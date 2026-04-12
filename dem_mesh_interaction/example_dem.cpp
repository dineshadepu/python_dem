/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/**
 * DEM Particle-Mesh Interaction using ArborX BVH
 *
 * Many particles fall under gravity onto a flat triangulated floor.
 * Only particle-mesh contact is modelled (no particle-particle).
 * Hertz normal force is computed per triangle via ArborX BVH sphere queries.
 * The mesh BVH is built once (static mesh) and queried every time step.
 *
 * Output:
 *   mesh.vtp                  - floor mesh (written once)
 *   particles_NNNNNN.vtp      - particle state at each output step
 *   particles.pvd             - ParaView time-series descriptor
 *
 * Usage:
 *   ./ArborX_Example_DEM_Mesh_Interaction.exe
 *       [--nx N]        particles per row in x        (default 16)
 *       [--ny N]        particles per row in y        (default 16)
 *       [--layers N]    particle layers stacked in z  (default 4)
 *       [--radius r]    particle radius [m]           (default 0.01)
 *       [--dt dt]       time step [s]                 (default 1e-5)
 *       [--steps N]     number of steps               (default 60000)
 *       [--out-every N] output interval               (default 1000)
 *       [--tank-l L]    tank side length [m]          (default 0.5)
 *       [--tank-h H]    tank wall height [m]          (default 0.5)
 *       [--mesh-n N]    mesh cells per unit length    (default 16)
 */

#include <ArborX.hpp>
#include <ArborX_Triangle.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// Type aliases
// ============================================================
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace    = ExecutionSpace::memory_space;
using HostSpace      = Kokkos::HostSpace;

using Point3d    = ArborX::Point<3, double>;
using Triangle3d = ArborX::Triangle<3, double>;
using Sphere3d   = ArborX::Sphere<3, double>;

// ============================================================
// AccessTraits: one sphere query per particle
// ============================================================
struct ParticleQueries
{
  Kokkos::View<Point3d *, MemorySpace> positions;
  Kokkos::View<double *,  MemorySpace> radii;
};

template <>
struct ArborX::AccessTraits<ParticleQueries>
{
  using memory_space = MemorySpace;

  static KOKKOS_FUNCTION std::size_t size(ParticleQueries const &q)
  {
    return q.positions.extent(0);
  }

  static KOKKOS_FUNCTION auto get(ParticleQueries const &q, std::size_t i)
  {
    // Search radius 1.5x particle radius — same buffer as the warp reference
    return ArborX::intersects(Sphere3d{q.positions(i), q.radii(i) * 1.5});
  }
};

// ============================================================
// Mesh contact force callback (Hertz normal force only)
//
// The 2-argument form is required when calling bvh.query() without
// output views.  For each (particle i, triangle j) pair found by the BVH
// traversal the callback:
//   1. Finds the closest point on the triangle to the particle centre.
//   2. Computes penetration depth delta = R - dist.
//   3. If delta > 0, applies Hertz normal force atomically to forces[i].
// ============================================================
struct MeshContactCallback
{
  Kokkos::View<Point3d *,    MemorySpace> positions;
  Kokkos::View<double *[3],  MemorySpace> forces;
  Kokkos::View<Triangle3d *, MemorySpace> triangles;
  Kokkos::View<double *,     MemorySpace> radii;
  double E_eff;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate const &pred,
                                   Value    const &val) const
  {
    int const i = static_cast<int>(ArborX::getData(pred)); // particle index
    int const j = val.index;                               // triangle index

    auto const &p   = positions(i);
    auto const &tri = triangles(j);

    // Closest point on this triangle to the particle centre
    auto cp = ArborX::Experimental::closestPoint(p, tri);

    double dx   = p[0] - cp[0];
    double dy   = p[1] - cp[1];
    double dz   = p[2] - cp[2];
    double dist = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

    double r       = radii(i);
    double overlap = r - dist;

    if (overlap <= 0.0 || dist < 1.0e-12)
      return;

    // Outward normal (mesh surface → particle centre)
    double inv_d = 1.0 / dist;
    double nx    = dx * inv_d;
    double ny    = dy * inv_d;
    double nz    = dz * inv_d;

    // Hertz: fn = (4/3) * E_eff * sqrt(R) * delta^(3/2)
    double fn = (4.0 / 3.0) * E_eff * Kokkos::sqrt(r)
                * overlap * Kokkos::sqrt(overlap);

    Kokkos::atomic_add(&forces(i, 0), fn * nx);
    Kokkos::atomic_add(&forces(i, 1), fn * ny);
    Kokkos::atomic_add(&forces(i, 2), fn * nz);
  }
};

// ============================================================
// Mesh generation: open-top 3-D tank  [0,L] x [0,L] x [0,H]
//
// Five faces: floor + left/right/front/back walls.
// fn = cells per side in x/y; fh = cells in z (scaled by H/L).
//
// Normal direction: because forces are computed as
//   n = (particle_centre - closest_point_on_tri) / dist
// the force always points away from whichever surface is closest —
// floor pushes +z, left wall pushes +x, right wall pushes -x, etc.
// No explicit winding or normal storage is needed.
// ============================================================
Kokkos::View<Triangle3d *, MemorySpace>
make_tank_triangles(double L, double H, int fn)
{
  // Vertical cell count proportional to aspect ratio so triangles stay
  // roughly square regardless of H/L.
  int fh = std::max(1, static_cast<int>(std::round(fn * H / L)));

  // 2 triangles per cell: floor fn^2, each wall fn*fh, 4 walls
  int total_tri = 2 * fn * fn + 4 * 2 * fn * fh;
  Kokkos::View<Triangle3d *, HostSpace> h_tri("host::tank", total_tri);

  int t = 0;

  // Bilinear interpolation on a planar quad defined by 4 corners.
  // (u,v) in [0,1]^2: u along p00→p10, v along p00→p01.
  auto bilin = [](Point3d p00, Point3d p10, Point3d p01, Point3d p11,
                  double u, double v) -> Point3d {
    double f00 = (1-u)*(1-v), f10 = u*(1-v), f01 = (1-u)*v, f11 = u*v;
    return Point3d{f00*p00[0] + f10*p10[0] + f01*p01[0] + f11*p11[0],
                   f00*p00[1] + f10*p10[1] + f01*p01[1] + f11*p11[1],
                   f00*p00[2] + f10*p10[2] + f01*p01[2] + f11*p11[2]};
  };

  // Tessellate one rectangular face into ncx * ncy * 2 triangles.
  auto add_face = [&](Point3d p00, Point3d p10, Point3d p01, Point3d p11,
                      int ncx, int ncy)
  {
    double hu = 1.0 / ncx, hv = 1.0 / ncy;
    for (int iy = 0; iy < ncy; ++iy)
      for (int ix = 0; ix < ncx; ++ix)
      {
        auto bl = bilin(p00, p10, p01, p11,  ix   *hu,  iy   *hv);
        auto br = bilin(p00, p10, p01, p11, (ix+1)*hu,  iy   *hv);
        auto tl = bilin(p00, p10, p01, p11,  ix   *hu, (iy+1)*hv);
        auto tr = bilin(p00, p10, p01, p11, (ix+1)*hu, (iy+1)*hv);
        h_tri(t++) = {bl, br, tr};
        h_tri(t++) = {bl, tr, tl};
      }
  };

  // Floor  (z = 0)
  add_face({0,0,0}, {L,0,0}, {0,L,0}, {L,L,0}, fn, fn);

  // Left wall   (x = 0,  y: 0→L,  z: 0→H)
  add_face({0,0,0}, {0,L,0}, {0,0,H}, {0,L,H}, fn, fh);

  // Right wall  (x = L,  y: 0→L,  z: 0→H)
  add_face({L,0,0}, {L,L,0}, {L,0,H}, {L,L,H}, fn, fh);

  // Front wall  (y = 0,  x: 0→L,  z: 0→H)
  add_face({0,0,0}, {L,0,0}, {0,0,H}, {L,0,H}, fn, fh);

  // Back wall   (y = L,  x: 0→L,  z: 0→H)
  add_face({0,L,0}, {L,L,0}, {0,L,H}, {L,L,H}, fn, fh);

  return Kokkos::create_mirror_view_and_copy(MemorySpace{}, h_tri);
}

// ============================================================
// VTK output helpers
// ============================================================

// Write particle positions, velocities and radii as a VTP PolyData file.
// Each particle is stored as a vertex ("Verts" cell type) so ParaView
// renders them as points; use "Glyph" filter + radius array for spheres.
void write_particles_vtp(
    const std::string                         &filename,
    Kokkos::View<Point3d *,   MemorySpace>    &d_pos,
    Kokkos::View<double *[3], MemorySpace>    &d_vel,
    Kokkos::View<double *,    MemorySpace>    &d_rad)
{
  int N = static_cast<int>(d_pos.extent(0));

  auto h_pos = Kokkos::create_mirror_view_and_copy(HostSpace{}, d_pos);
  auto h_vel = Kokkos::create_mirror_view_and_copy(HostSpace{}, d_vel);
  auto h_rad = Kokkos::create_mirror_view_and_copy(HostSpace{}, d_rad);

  std::ofstream f(filename);
  f << std::scientific << std::setprecision(6);

  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <PolyData>\n"
    << "    <Piece NumberOfPoints=\"" << N << "\""
    << " NumberOfVerts=\"" << N << "\""
    << " NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

  // ---- Points ----
  f << "      <Points>\n"
    << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
    << " format=\"ascii\">\n";
  for (int i = 0; i < N; ++i)
    f << "          " << h_pos(i)[0] << " "
                      << h_pos(i)[1] << " "
                      << h_pos(i)[2] << "\n";
  f << "        </DataArray>\n"
    << "      </Points>\n";

  // ---- Verts (one vertex cell per particle) ----
  f << "      <Verts>\n"
    << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n"
    << "         ";
  for (int i = 0; i < N; ++i) f << " " << i;
  f << "\n        </DataArray>\n"
    << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n"
    << "         ";
  for (int i = 1; i <= N; ++i) f << " " << i;
  f << "\n        </DataArray>\n"
    << "      </Verts>\n";

  // ---- PointData ----
  f << "      <PointData>\n";

  // velocity vector
  f << "        <DataArray type=\"Float64\" Name=\"velocity\""
    << " NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int i = 0; i < N; ++i)
    f << "          " << h_vel(i, 0) << " "
                      << h_vel(i, 1) << " "
                      << h_vel(i, 2) << "\n";
  f << "        </DataArray>\n";

  // radius scalar (for Glyph filter in ParaView)
  f << "        <DataArray type=\"Float64\" Name=\"radius\""
    << " format=\"ascii\">\n";
  for (int i = 0; i < N; ++i)
    f << "          " << h_rad(i) << "\n";
  f << "        </DataArray>\n";

  f << "      </PointData>\n"
    << "    </Piece>\n"
    << "  </PolyData>\n"
    << "</VTKFile>\n";
}

// Write the triangulated mesh as a VTP PolyData file (call once).
void write_mesh_vtp(
    const std::string                          &filename,
    Kokkos::View<Triangle3d *, MemorySpace>    &d_tri)
{
  int num_tri = static_cast<int>(d_tri.extent(0));
  auto h_tri  = Kokkos::create_mirror_view_and_copy(HostSpace{}, d_tri);

  int num_pts = 3 * num_tri; // every triangle stores its 3 vertices

  std::ofstream f(filename);
  f << std::scientific << std::setprecision(6);

  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <PolyData>\n"
    << "    <Piece NumberOfPoints=\"" << num_pts << "\""
    << " NumberOfVerts=\"0\" NumberOfLines=\"0\""
    << " NumberOfStrips=\"0\" NumberOfPolys=\"" << num_tri << "\">\n";

  // ---- Points ----
  f << "      <Points>\n"
    << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
    << " format=\"ascii\">\n";
  for (int t = 0; t < num_tri; ++t)
  {
    auto const &tri = h_tri(t);
    f << "          " << tri.a[0] << " " << tri.a[1] << " " << tri.a[2] << "\n";
    f << "          " << tri.b[0] << " " << tri.b[1] << " " << tri.b[2] << "\n";
    f << "          " << tri.c[0] << " " << tri.c[1] << " " << tri.c[2] << "\n";
  }
  f << "        </DataArray>\n"
    << "      </Points>\n";

  // ---- Polys ----
  f << "      <Polys>\n"
    << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int t = 0; t < num_tri; ++t)
    f << "          " << 3*t << " " << 3*t+1 << " " << 3*t+2 << "\n";
  f << "        </DataArray>\n"
    << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  for (int t = 1; t <= num_tri; ++t)
    f << "          " << 3*t << "\n";
  f << "        </DataArray>\n"
    << "      </Polys>\n"
    << "    </Piece>\n"
    << "  </PolyData>\n"
    << "</VTKFile>\n";
}

// Write a ParaView Data (PVD) file that lists all particle VTP files with
// their corresponding simulation times.  Call once after the loop.
void write_pvd(
    const std::string                              &filename,
    const std::vector<std::pair<double,std::string>> &entries)
{
  std::ofstream f(filename);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"Collection\" version=\"0.1\""
    << " byte_order=\"LittleEndian\">\n"
    << "  <Collection>\n";
  for (auto const &[time, vtp] : entries)
    f << "    <DataSet timestep=\"" << time
      << "\" group=\"\" part=\"0\" file=\"" << vtp << "\"/>\n";
  f << "  </Collection>\n"
    << "</VTKFile>\n";
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  // ---- default parameters ----
  int    nx        = 16;     // particles per row in x
  int    ny        = 16;     // particles per row in y
  int    layers    = 4;      // particle layers stacked in z
  double radius    = 0.01;   // particle radius [m]
  double rho       = 2500.0; // particle density [kg/m^3]
  double E_p       = 1.0e7;  // particle Young's modulus [Pa]
  double nu_p      = 0.3;    // particle Poisson ratio
  double E_w       = 1.0e7;  // wall Young's modulus [Pa]
  double nu_w      = 0.3;    // wall Poisson ratio
  double gz        = -9.81;  // gravity [m/s^2]
  double dt        = 1.0e-5; // time step [s]
  int    steps     = 60000;  // number of steps
  int    out_every = 1000;   // output interval
  double L         = 0.5;    // tank side length [m]
  double H         = 0.5;    // tank wall height [m]
  int    fn        = 16;     // mesh cells per unit length (x/y); z scales with H/L

  // ---- parse arguments ----
  for (int a = 1; a < argc; ++a)
  {
    std::string arg(argv[a]);
    if      (arg == "--nx"        && a+1 < argc) nx        = std::atoi(argv[++a]);
    else if (arg == "--ny"        && a+1 < argc) ny        = std::atoi(argv[++a]);
    else if (arg == "--layers"    && a+1 < argc) layers    = std::atoi(argv[++a]);
    else if (arg == "--radius"    && a+1 < argc) radius    = std::atof(argv[++a]);
    else if (arg == "--dt"        && a+1 < argc) dt        = std::atof(argv[++a]);
    else if (arg == "--steps"     && a+1 < argc) steps     = std::atoi(argv[++a]);
    else if (arg == "--out-every" && a+1 < argc) out_every = std::atoi(argv[++a]);
    else if (arg == "--tank-l"    && a+1 < argc) L         = std::atof(argv[++a]);
    else if (arg == "--tank-h"    && a+1 < argc) H         = std::atof(argv[++a]);
    else if (arg == "--mesh-n"    && a+1 < argc) fn        = std::atoi(argv[++a]);
    else if (arg == "--help")
    {
      std::cout
        << "Usage: " << argv[0] << "\n"
        << "  --nx N        particles per row in x   (default 16)\n"
        << "  --ny N        particles per row in y   (default 16)\n"
        << "  --layers N    layers stacked in z       (default 4)\n"
        << "  --radius r    particle radius [m]       (default 0.01)\n"
        << "  --dt dt       time step [s]             (default 1e-5)\n"
        << "  --steps N     total steps               (default 60000)\n"
        << "  --out-every N output interval           (default 1000)\n"
        << "  --tank-l L    tank side length [m]      (default 0.5)\n"
        << "  --tank-h H    tank wall height [m]      (default 0.5)\n"
        << "  --mesh-n N    mesh cells per unit len   (default 16)\n";
      return 0;
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      return EXIT_FAILURE;
    }
  }

  int    N    = nx * ny * layers;
  double E_eff = 1.0 / ((1.0 - nu_p*nu_p)/E_p + (1.0 - nu_w*nu_w)/E_w);
  double m_p   = (4.0/3.0) * M_PI * radius*radius*radius * rho;
  double spacing = 2.2 * radius; // centre-to-centre distance in the grid

  std::cout << "DEM Particle-Mesh Interaction (Hertz normal, no p-p contact)\n";
  std::cout << "  Backend:         " << ExecutionSpace().name() << "\n";
  std::cout << "  Particles:       " << N
            << "  (" << nx << " x " << ny << " x " << layers << " layers)\n";
  std::cout << "  Radius:          " << radius    << " m\n";
  std::cout << "  E_eff:           " << E_eff     << " Pa\n";
  std::cout << "  dt:              " << dt        << " s\n";
  std::cout << "  Steps:           " << steps     << "\n";
  std::cout << "  Output every:    " << out_every << " steps\n";
  std::cout << "  Tank:            " << L << " x " << L << " x " << H << " m\n";

  // ---- build static mesh BVH (once) ----
  // Tank = floor (z=0) + 4 vertical walls (x=0, x=L, y=0, y=L)
  auto triangles = make_tank_triangles(L, H, fn);
  std::cout << "  Tank triangles:  " << triangles.extent(0) << "\n\n";

  ExecutionSpace exec_space;
  ArborX::BoundingVolumeHierarchy bvh{
      exec_space, ArborX::Experimental::attach_indices(triangles)};

  // Write the tank mesh once (static geometry)
  write_mesh_vtp("tank.vtp", triangles);
  std::cout << "Wrote tank.vtp\n";

  // ---- particle state views ----
  Kokkos::View<Point3d *,   MemorySpace> positions ("positions",  N);
  Kokkos::View<double *[3], MemorySpace> velocities("velocities", N);
  Kokkos::View<double *[3], MemorySpace> forces    ("forces",     N);
  Kokkos::View<double *,    MemorySpace> radii     ("radii",      N);
  Kokkos::View<double *,    MemorySpace> masses    ("masses",     N);

  // ---- initialise particles: nx*ny grid per layer, layers stacked in z ----
  {
    auto h_pos = Kokkos::create_mirror_view(positions);
    auto h_vel = Kokkos::create_mirror_view(velocities);
    auto h_rad = Kokkos::create_mirror_view(radii);
    auto h_mas = Kokkos::create_mirror_view(masses);

    // Offset from the floor edge so the grid fits inside [0, L]
    double x0 = (L - (nx - 1) * spacing) * 0.5;
    double y0 = (L - (ny - 1) * spacing) * 0.5;
    // First layer starts just above contact range; layers stack upward
    double z0 = 5.0 * radius;

    int idx = 0;
    for (int k = 0; k < layers; ++k)
    {
      // Offset alternate layers by half a spacing for a more natural packing
      double xoff = (k % 2 == 1) ? 0.5 * spacing : 0.0;
      double yoff = (k % 2 == 1) ? 0.5 * spacing : 0.0;
      double z    = z0 + k * 2.5 * radius;

      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          h_pos(idx) = Point3d{x0 + i*spacing + xoff,
                               y0 + j*spacing + yoff,
                               z};
          h_vel(idx, 0) = 0.0;
          h_vel(idx, 1) = 0.0;
          h_vel(idx, 2) = 0.0;
          h_rad(idx)    = radius;
          h_mas(idx)    = m_p;
          ++idx;
        }
    }

    Kokkos::deep_copy(positions,  h_pos);
    Kokkos::deep_copy(velocities, h_vel);
    Kokkos::deep_copy(radii,      h_rad);
    Kokkos::deep_copy(masses,     h_mas);
  }

  // ---- initialise forces (gravity only, used in first half-step) ----
  Kokkos::parallel_for(
      "init::forces",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
      KOKKOS_LAMBDA(int i) {
        forces(i, 0) = 0.0;
        forces(i, 1) = 0.0;
        forces(i, 2) = masses(i) * gz;
      });

  // ---- PVD entries collected during the run ----
  std::vector<std::pair<double, std::string>> pvd_entries;

  // ---- helper: output filename for step s ----
  auto particle_filename = [](int s) {
    std::ostringstream ss;
    ss << "particles_" << std::setw(6) << std::setfill('0') << s << ".vtp";
    return ss.str();
  };

  std::cout << "step, time [s], z_min [m], z_avg [m], z_max [m]\n";

  // ============================================================
  // Velocity Verlet time loop
  //   1. v += 0.5*dt * f/m          (half-step, old forces)
  //   2. x += dt * v                (update positions)
  //   3. f = gravity + contact(x)   (new forces)
  //   4. v += 0.5*dt * f/m          (half-step, new forces)
  // ============================================================
  for (int step = 0; step <= steps; ++step)
  {
    double time = step * dt;

    // --- output ---
    if (step % out_every == 0)
    {
      // Stats to stdout
      auto h_pos = Kokkos::create_mirror_view_and_copy(HostSpace{}, positions);
      double z_min = h_pos(0)[2], z_max = h_pos(0)[2], z_avg = 0.0;
      for (int i = 0; i < N; ++i)
      {
        double z = h_pos(i)[2];
        z_avg += z;
        if (z < z_min) z_min = z;
        if (z > z_max) z_max = z;
      }
      z_avg /= N;
      std::cout << step << ", " << time << ", "
                << z_min << ", " << z_avg << ", " << z_max << "\n";

      // VTP file
      std::string fname = particle_filename(step);
      write_particles_vtp(fname, positions, velocities, radii);
      pvd_entries.emplace_back(time, fname);
    }

    if (step == steps) break;

    // --- stage 1: half-step velocity (old forces) ---
    Kokkos::parallel_for(
        "vv::half1",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          double inv_m = 1.0 / masses(i);
          velocities(i, 0) += 0.5 * dt * forces(i, 0) * inv_m;
          velocities(i, 1) += 0.5 * dt * forces(i, 1) * inv_m;
          velocities(i, 2) += 0.5 * dt * forces(i, 2) * inv_m;
        });

    // --- stage 2: update positions ---
    Kokkos::parallel_for(
        "vv::pos",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          positions(i)[0] += dt * velocities(i, 0);
          positions(i)[1] += dt * velocities(i, 1);
          positions(i)[2] += dt * velocities(i, 2);
        });

    // --- stage 3: reset forces + gravity ---
    Kokkos::parallel_for(
        "vv::gravity",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          forces(i, 0) = 0.0;
          forces(i, 1) = 0.0;
          forces(i, 2) = masses(i) * gz;
        });

    // --- stage 4: mesh contact forces via BVH ---
    // BVH is built once; only queries (sphere centres) change each step.
    // MeshContactCallback accumulates Hertz normal forces atomically.
    bvh.query(exec_space,
              ArborX::Experimental::attach_indices<int>(
                  ParticleQueries{positions, radii}),
              MeshContactCallback{positions, forces, triangles, radii, E_eff});

    // --- stage 5: half-step velocity (new forces) ---
    Kokkos::parallel_for(
        "vv::half2",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          double inv_m = 1.0 / masses(i);
          velocities(i, 0) += 0.5 * dt * forces(i, 0) * inv_m;
          velocities(i, 1) += 0.5 * dt * forces(i, 1) * inv_m;
          velocities(i, 2) += 0.5 * dt * forces(i, 2) * inv_m;
        });
  }

  Kokkos::fence();

  // ---- write PVD time-series descriptor ----
  write_pvd("particles.pvd", pvd_entries);
  std::cout << "\nWrote " << pvd_entries.size() << " VTP files + particles.pvd\n";
  std::cout << "Open particles.pvd in ParaView to view the time series.\n";
  std::cout << "Simulation complete.\n";
  return 0;
}
