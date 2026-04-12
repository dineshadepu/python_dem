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
 * DEM Particle-Tank Interaction — Accurate Sphere-Triangle Contact
 *
 * Improved contact model based on the DEM-Engine collision kernels
 * (DEMCollisionKernels.cu, Ericson 2005 "Real-time Collision Detection").
 *
 * Contact geometry (per triangle):
 *   1.  snap_to_face  — find the closest point Q on the triangle to the
 *       sphere centre P and report whether Q lies on the face interior
 *       or on an edge / vertex.
 *   2a. Face contact  (Q inside triangle):
 *         normal  = ±face_n  (cross-product normal, sign chosen so it
 *                              points from the surface toward the sphere)
 *         overlap = radius - |h|   where h = signed height above face plane
 *   2b. Edge/vertex contact  (Q on boundary):
 *         normal  = (P - Q) / |P - Q|
 *         overlap = radius - |P - Q|
 *         guard   : |h| < radius  (sphere must be near the face plane —
 *                   prevents false positives from extended edge regions)
 *
 * The distinction matters at corners and edges shared between triangles:
 * with only a closest-point direction (old approach) the normal "rotates"
 * continuously across the edge, giving a non-physical tangential component.
 * With the face-normal approach the force is always perpendicular to the
 * flat face when the sphere overlaps the face interior, matching physical
 * expectation.
 *
 * Outputs (same as example_dem):
 *   tank.vtp                  - tank geometry (floor + 4 walls)
 *   particles_NNNNNN.vtp      - particle state at each output step
 *   particles.pvd             - ParaView time-series descriptor
 *
 * Usage:
 *   ./ArborX_Example_DEM_Triangle_Contact.exe
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
// snap_to_face
//
// Finds the closest point Q on the triangle ABC to point P.
// Writes Q into (qx,qy,qz).
// Returns TRUE  if Q lies on an edge or vertex (boundary contact).
// Returns FALSE if Q lies strictly inside the face (face contact).
//
// Algorithm: Ericson, "Real-Time Collision Detection", 2005, pp. 141.
// Ported from DEM-Engine DEMCollisionKernels.cu with CUDA intrinsics
// replaced by standard double arithmetic.
// ============================================================
KOKKOS_INLINE_FUNCTION
bool snap_to_face(double ax, double ay, double az,
                  double bx, double by, double bz,
                  double cx, double cy, double cz,
                  double px, double py, double pz,
                  double &qx, double &qy, double &qz)
{
  const double abx = bx-ax, aby = by-ay, abz = bz-az;
  const double acx = cx-ax, acy = cy-ay, acz = cz-az;
  const double apx = px-ax, apy = py-ay, apz = pz-az;

  const double d1 = abx*apx + aby*apy + abz*apz;  // dot(AB, AP)
  const double d2 = acx*apx + acy*apy + acz*apz;  // dot(AC, AP)
  if (d1 <= 0.0 && d2 <= 0.0) { qx=ax; qy=ay; qz=az; return true; }  // A

  const double bpx = px-bx, bpy = py-by, bpz = pz-bz;
  const double d3 = abx*bpx + aby*bpy + abz*bpz;  // dot(AB, BP)
  const double d4 = acx*bpx + acy*bpy + acz*bpz;  // dot(AC, BP)
  if (d3 >= 0.0 && d4 <= d3) { qx=bx; qy=by; qz=bz; return true; }  // B

  const double vc = d1*d4 - d3*d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {       // edge AB
    const double v = d1 / (d1 - d3);
    qx = ax + v*abx; qy = ay + v*aby; qz = az + v*abz;
    return true;
  }

  const double cpx = px-cx, cpy = py-cy, cpz = pz-cz;
  const double d5 = abx*cpx + aby*cpy + abz*cpz;  // dot(AB, CP)
  const double d6 = acx*cpx + acy*cpy + acz*cpz;  // dot(AC, CP)
  if (d6 >= 0.0 && d5 <= d6) { qx=cx; qy=cy; qz=cz; return true; }  // C

  const double vb = d5*d2 - d1*d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {       // edge AC
    const double w = d2 / (d2 - d6);
    qx = ax + w*acx; qy = ay + w*acy; qz = az + w*acz;
    return true;
  }

  const double va = d3*d6 - d5*d4;
  if (va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0) {  // edge BC
    const double w = (d4-d3) / ((d4-d3) + (d5-d6));
    qx = bx + w*(cx-bx); qy = by + w*(cy-by); qz = bz + w*(cz-bz);
    return true;
  }

  // P projects inside the face
  const double denom = 1.0 / (va + vb + vc);
  const double v     = vb * denom;
  const double w     = vc * denom;
  qx = ax + v*abx + w*acx;
  qy = ay + v*aby + w*acy;
  qz = az + v*abz + w*acz;
  return false;
}

// ============================================================
// triangle_sphere_contact
//
// Accurate sphere-triangle contact detection.
//
// Parameters:
//   (ax..cz)  Triangle vertices A, B, C
//   (px..pz)  Sphere centre
//   radius    Sphere radius
//
// Outputs (written only if contact detected):
//   (nx,ny,nz)  Contact normal pointing FROM triangle surface TOWARD sphere.
//   overlap     Penetration depth (> 0 when in contact).
//
// Returns true if and only if the sphere overlaps the triangle.
//
// Logic (mirrors DEM-Engine triangle_sphere_CD):
//   - face normal  fn = normalize(cross(B-A, C-A))
//   - signed height  h = dot(P-A, fn)
//   - Q = snap_to_face(...)
//   Face contact  (Q inside triangle):
//       overlap = radius - |h|
//       normal  = sign(h) * fn     <- geometric, perpendicular to face
//   Edge/vertex contact  (Q on boundary):
//       dist    = |P - Q|
//       overlap = radius - dist
//       guard   : |h| < radius     (sphere near face plane)
//       normal  = (P - Q) / dist   <- points away from closest boundary
// ============================================================
KOKKOS_INLINE_FUNCTION
bool triangle_sphere_contact(
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double px, double py, double pz,
    double radius,
    double &nx, double &ny, double &nz,
    double &overlap)
{
  // ---- face normal (right-hand rule) ----
  const double abx = bx-ax, aby = by-ay, abz = bz-az;
  const double acx = cx-ax, acy = cy-ay, acz = cz-az;
  double fnx = aby*acz - abz*acy;
  double fny = abz*acx - abx*acz;
  double fnz = abx*acy - aby*acx;
  const double fn_len = Kokkos::sqrt(fnx*fnx + fny*fny + fnz*fnz);
  if (fn_len < 1.0e-14) return false;  // degenerate triangle
  const double inv_fn = 1.0 / fn_len;
  fnx *= inv_fn; fny *= inv_fn; fnz *= inv_fn;

  // ---- signed height of sphere centre above face plane ----
  const double h     = (px-ax)*fnx + (py-ay)*fny + (pz-az)*fnz;
  const double abs_h = Kokkos::fabs(h);

  // ---- closest point on triangle ----
  double qx, qy, qz;
  const bool on_edge = snap_to_face(ax,ay,az, bx,by,bz, cx,cy,cz,
                                    px,py,pz, qx,qy,qz);

  if (!on_edge) {
    // ---- face contact ----
    overlap = radius - abs_h;
    if (overlap <= 0.0) return false;

    // Normal from surface toward sphere; sign follows signed height
    const double s = (h >= 0.0) ? 1.0 : -1.0;
    nx = s * fnx; ny = s * fny; nz = s * fnz;

  } else {
    // ---- edge / vertex contact ----
    const double dx   = px - qx;
    const double dy   = py - qy;
    const double dz   = pz - qz;
    const double dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    if (dist < 1.0e-12) return false;

    overlap = radius - dist;
    if (overlap <= 0.0) return false;

    // Guard: sphere must be near the face plane.
    // Without this, a sphere sitting far above an edge of an adjacent
    // triangle would register a spurious contact with that triangle's
    // extended edge.  This mirrors the h-check in DEM-Engine's
    // triangle_sphere_CD().
    if (abs_h >= radius) return false;

    const double inv_d = 1.0 / dist;
    nx = dx * inv_d; ny = dy * inv_d; nz = dz * inv_d;
  }
  return true;
}

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
    return ArborX::intersects(Sphere3d{q.positions(i), q.radii(i) * 1.5});
  }
};

// ============================================================
// Mesh contact force callback — accurate sphere-triangle contact
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
    const int i = static_cast<int>(ArborX::getData(pred)); // particle index
    const int j = val.index;                               // triangle index

    auto const &p   = positions(i);
    auto const &tri = triangles(j);
    const double r  = radii(i);

    double nx, ny, nz, overlap;
    const bool in_contact = triangle_sphere_contact(
        tri.a[0], tri.a[1], tri.a[2],
        tri.b[0], tri.b[1], tri.b[2],
        tri.c[0], tri.c[1], tri.c[2],
        p[0], p[1], p[2],
        r, nx, ny, nz, overlap);

    if (!in_contact) return;

    // Hertz normal force: fn = (4/3) * E_eff * sqrt(R) * overlap^(3/2)
    const double fn = (4.0/3.0) * E_eff * Kokkos::sqrt(r)
                      * overlap * Kokkos::sqrt(overlap);

    Kokkos::atomic_add(&forces(i, 0), fn * nx);
    Kokkos::atomic_add(&forces(i, 1), fn * ny);
    Kokkos::atomic_add(&forces(i, 2), fn * nz);
  }
};

// ============================================================
// Mesh generation: open-top tank  [0,L] x [0,L] x [0,H]
// ============================================================
Kokkos::View<Triangle3d *, MemorySpace>
make_tank_triangles(double L, double H, int fn)
{
  const int fh        = std::max(1, static_cast<int>(std::round(fn * H / L)));
  const int total_tri = 2 * fn * fn + 4 * 2 * fn * fh;
  Kokkos::View<Triangle3d *, HostSpace> h_tri("host::tank", total_tri);

  int t = 0;

  auto bilin = [](Point3d p00, Point3d p10, Point3d p01, Point3d p11,
                  double u, double v) -> Point3d {
    const double f00=(1-u)*(1-v), f10=u*(1-v), f01=(1-u)*v, f11=u*v;
    return Point3d{f00*p00[0]+f10*p10[0]+f01*p01[0]+f11*p11[0],
                   f00*p00[1]+f10*p10[1]+f01*p01[1]+f11*p11[1],
                   f00*p00[2]+f10*p10[2]+f01*p01[2]+f11*p11[2]};
  };

  auto add_face = [&](Point3d p00, Point3d p10, Point3d p01, Point3d p11,
                      int ncx, int ncy)
  {
    const double hu = 1.0/ncx, hv = 1.0/ncy;
    for (int iy = 0; iy < ncy; ++iy)
      for (int ix = 0; ix < ncx; ++ix)
      {
        auto bl = bilin(p00,p10,p01,p11,  ix   *hu,  iy   *hv);
        auto br = bilin(p00,p10,p01,p11, (ix+1)*hu,  iy   *hv);
        auto tl = bilin(p00,p10,p01,p11,  ix   *hu, (iy+1)*hv);
        auto tr = bilin(p00,p10,p01,p11, (ix+1)*hu, (iy+1)*hv);
        h_tri(t++) = {bl, br, tr};
        h_tri(t++) = {bl, tr, tl};
      }
  };

  add_face({0,0,0},{L,0,0},{0,L,0},{L,L,0}, fn, fn); // floor   z=0
  add_face({0,0,0},{0,L,0},{0,0,H},{0,L,H}, fn, fh); // left    x=0
  add_face({L,0,0},{L,L,0},{L,0,H},{L,L,H}, fn, fh); // right   x=L
  add_face({0,0,0},{L,0,0},{0,0,H},{L,0,H}, fn, fh); // front   y=0
  add_face({0,L,0},{L,L,0},{0,L,H},{L,L,H}, fn, fh); // back    y=L

  return Kokkos::create_mirror_view_and_copy(MemorySpace{}, h_tri);
}

// ============================================================
// VTK output helpers (identical to example_dem.cpp)
// ============================================================
void write_particles_vtp(
    const std::string                         &filename,
    Kokkos::View<Point3d *,   MemorySpace>    &d_pos,
    Kokkos::View<double *[3], MemorySpace>    &d_vel,
    Kokkos::View<double *,    MemorySpace>    &d_rad)
{
  const int N = static_cast<int>(d_pos.extent(0));
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
    << " NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n"
    << "      <Points>\n"
    << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int i = 0; i < N; ++i)
    f << "          " << h_pos(i)[0] << " " << h_pos(i)[1] << " " << h_pos(i)[2] << "\n";
  f << "        </DataArray>\n      </Points>\n"
    << "      <Verts>\n"
    << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n         ";
  for (int i = 0; i < N; ++i) f << " " << i;
  f << "\n        </DataArray>\n"
    << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n         ";
  for (int i = 1; i <= N; ++i) f << " " << i;
  f << "\n        </DataArray>\n      </Verts>\n"
    << "      <PointData>\n"
    << "        <DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int i = 0; i < N; ++i)
    f << "          " << h_vel(i,0) << " " << h_vel(i,1) << " " << h_vel(i,2) << "\n";
  f << "        </DataArray>\n"
    << "        <DataArray type=\"Float64\" Name=\"radius\" format=\"ascii\">\n";
  for (int i = 0; i < N; ++i) f << "          " << h_rad(i) << "\n";
  f << "        </DataArray>\n      </PointData>\n"
    << "    </Piece>\n  </PolyData>\n</VTKFile>\n";
}

void write_mesh_vtp(
    const std::string                          &filename,
    Kokkos::View<Triangle3d *, MemorySpace>    &d_tri)
{
  const int num_tri = static_cast<int>(d_tri.extent(0));
  auto h_tri = Kokkos::create_mirror_view_and_copy(HostSpace{}, d_tri);
  const int num_pts = 3 * num_tri;

  std::ofstream f(filename);
  f << std::scientific << std::setprecision(6);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <PolyData>\n"
    << "    <Piece NumberOfPoints=\"" << num_pts << "\""
    << " NumberOfVerts=\"0\" NumberOfLines=\"0\""
    << " NumberOfStrips=\"0\" NumberOfPolys=\"" << num_tri << "\">\n"
    << "      <Points>\n"
    << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int t = 0; t < num_tri; ++t)
  {
    auto const &tri = h_tri(t);
    f << "          " << tri.a[0] << " " << tri.a[1] << " " << tri.a[2] << "\n"
      << "          " << tri.b[0] << " " << tri.b[1] << " " << tri.b[2] << "\n"
      << "          " << tri.c[0] << " " << tri.c[1] << " " << tri.c[2] << "\n";
  }
  f << "        </DataArray>\n      </Points>\n"
    << "      <Polys>\n"
    << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int t = 0; t < num_tri; ++t)
    f << "          " << 3*t << " " << 3*t+1 << " " << 3*t+2 << "\n";
  f << "        </DataArray>\n"
    << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  for (int t = 1; t <= num_tri; ++t) f << "          " << 3*t << "\n";
  f << "        </DataArray>\n      </Polys>\n"
    << "    </Piece>\n  </PolyData>\n</VTKFile>\n";
}

void write_pvd(
    const std::string                                   &filename,
    const std::vector<std::pair<double,std::string>>    &entries)
{
  std::ofstream f(filename);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <Collection>\n";
  for (auto const &[time, vtp] : entries)
    f << "    <DataSet timestep=\"" << time
      << "\" group=\"\" part=\"0\" file=\"" << vtp << "\"/>\n";
  f << "  </Collection>\n</VTKFile>\n";
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  // ---- default parameters ----
  int    nx        = 16;
  int    ny        = 16;
  int    layers    = 4;
  double radius    = 0.01;
  double rho       = 2500.0;
  double E_p       = 1.0e7;
  double nu_p      = 0.3;
  double E_w       = 1.0e7;
  double nu_w      = 0.3;
  double gz        = -9.81;
  double dt        = 1.0e-5;
  int    steps     = 60000;
  int    out_every = 1000;
  double L         = 0.5;
  double H         = 0.5;
  int    fn        = 16;

  for (int a = 1; a < argc; ++a)
  {
    std::string arg(argv[a]);
    if      (arg=="--nx"        && a+1<argc) nx        = std::atoi(argv[++a]);
    else if (arg=="--ny"        && a+1<argc) ny        = std::atoi(argv[++a]);
    else if (arg=="--layers"    && a+1<argc) layers    = std::atoi(argv[++a]);
    else if (arg=="--radius"    && a+1<argc) radius    = std::atof(argv[++a]);
    else if (arg=="--dt"        && a+1<argc) dt        = std::atof(argv[++a]);
    else if (arg=="--steps"     && a+1<argc) steps     = std::atoi(argv[++a]);
    else if (arg=="--out-every" && a+1<argc) out_every = std::atoi(argv[++a]);
    else if (arg=="--tank-l"    && a+1<argc) L         = std::atof(argv[++a]);
    else if (arg=="--tank-h"    && a+1<argc) H         = std::atof(argv[++a]);
    else if (arg=="--mesh-n"    && a+1<argc) fn        = std::atoi(argv[++a]);
    else if (arg=="--help")
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
    else { std::cerr << "Unknown argument: " << arg << "\n"; return EXIT_FAILURE; }
  }

  const int    N     = nx * ny * layers;
  const double E_eff = 1.0 / ((1.0-nu_p*nu_p)/E_p + (1.0-nu_w*nu_w)/E_w);
  const double m_p   = (4.0/3.0) * M_PI * radius*radius*radius * rho;
  const double spacing = 2.2 * radius;

  std::cout << "DEM Particle-Tank (accurate sphere-triangle contact)\n";
  std::cout << "  Backend:         " << ExecutionSpace().name() << "\n";
  std::cout << "  Particles:       " << N
            << "  (" << nx << " x " << ny << " x " << layers << " layers)\n";
  std::cout << "  Radius:          " << radius << " m\n";
  std::cout << "  E_eff:           " << E_eff  << " Pa\n";
  std::cout << "  dt:              " << dt     << " s\n";
  std::cout << "  Steps:           " << steps  << "\n";
  std::cout << "  Tank:            " << L << " x " << L << " x " << H << " m\n";

  // ---- build static mesh BVH (once) ----
  auto triangles = make_tank_triangles(L, H, fn);
  std::cout << "  Tank triangles:  " << triangles.extent(0) << "\n\n";

  ExecutionSpace exec_space;
  ArborX::BoundingVolumeHierarchy bvh{
      exec_space, ArborX::Experimental::attach_indices(triangles)};

  write_mesh_vtp("tank.vtp", triangles);
  std::cout << "Wrote tank.vtp\n";

  // ---- particle state views ----
  Kokkos::View<Point3d *,   MemorySpace> positions ("positions",  N);
  Kokkos::View<double *[3], MemorySpace> velocities("velocities", N);
  Kokkos::View<double *[3], MemorySpace> forces    ("forces",     N);
  Kokkos::View<double *,    MemorySpace> radii     ("radii",      N);
  Kokkos::View<double *,    MemorySpace> masses    ("masses",     N);

  // ---- initialise particles in staggered grid above floor ----
  {
    auto h_pos = Kokkos::create_mirror_view(positions);
    auto h_vel = Kokkos::create_mirror_view(velocities);
    auto h_rad = Kokkos::create_mirror_view(radii);
    auto h_mas = Kokkos::create_mirror_view(masses);

    const double x0 = (L - (nx-1)*spacing) * 0.5;
    const double y0 = (L - (ny-1)*spacing) * 0.5;
    const double z0 = 5.0 * radius;

    int idx = 0;
    for (int k = 0; k < layers; ++k)
    {
      const double xoff = (k%2==1) ? 0.5*spacing : 0.0;
      const double yoff = (k%2==1) ? 0.5*spacing : 0.0;
      const double z    = z0 + k * 2.5 * radius;

      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          h_pos(idx) = Point3d{x0 + i*spacing + xoff,
                               y0 + j*spacing + yoff, z};
          h_vel(idx,0)=0.0; h_vel(idx,1)=0.0; h_vel(idx,2)=0.0;
          h_rad(idx) = radius;
          h_mas(idx) = m_p;
          ++idx;
        }
    }
    Kokkos::deep_copy(positions,  h_pos);
    Kokkos::deep_copy(velocities, h_vel);
    Kokkos::deep_copy(radii,      h_rad);
    Kokkos::deep_copy(masses,     h_mas);
  }

  // ---- initialise forces (gravity, used in first half-step) ----
  Kokkos::parallel_for(
      "init::forces",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
      KOKKOS_LAMBDA(int i) {
        forces(i,0)=0.0; forces(i,1)=0.0; forces(i,2)=masses(i)*gz;
      });

  std::vector<std::pair<double,std::string>> pvd_entries;

  auto particle_filename = [](int s) {
    std::ostringstream ss;
    ss << "particles_" << std::setw(6) << std::setfill('0') << s << ".vtp";
    return ss.str();
  };

  std::cout << "step, time [s], z_min [m], z_avg [m], z_max [m]\n";

  // ============================================================
  // Velocity Verlet time loop
  // ============================================================
  for (int step = 0; step <= steps; ++step)
  {
    const double time = step * dt;

    if (step % out_every == 0)
    {
      auto h_pos = Kokkos::create_mirror_view_and_copy(HostSpace{}, positions);
      double z_min=h_pos(0)[2], z_max=h_pos(0)[2], z_avg=0.0;
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

      std::string fname = particle_filename(step);
      write_particles_vtp(fname, positions, velocities, radii);
      pvd_entries.emplace_back(time, fname);
    }

    if (step == steps) break;

    // stage 1: half-step velocity
    Kokkos::parallel_for(
        "vv::half1", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          const double inv_m = 1.0 / masses(i);
          velocities(i,0) += 0.5*dt*forces(i,0)*inv_m;
          velocities(i,1) += 0.5*dt*forces(i,1)*inv_m;
          velocities(i,2) += 0.5*dt*forces(i,2)*inv_m;
        });

    // stage 2: update positions
    Kokkos::parallel_for(
        "vv::pos", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          positions(i)[0] += dt*velocities(i,0);
          positions(i)[1] += dt*velocities(i,1);
          positions(i)[2] += dt*velocities(i,2);
        });

    // stage 3: reset forces + gravity
    Kokkos::parallel_for(
        "vv::gravity", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          forces(i,0)=0.0; forces(i,1)=0.0; forces(i,2)=masses(i)*gz;
        });

    // stage 4: mesh contact forces — accurate sphere-triangle detection
    bvh.query(exec_space,
              ArborX::Experimental::attach_indices<int>(
                  ParticleQueries{positions, radii}),
              MeshContactCallback{positions, forces, triangles, radii, E_eff});

    // stage 5: second half-step velocity
    Kokkos::parallel_for(
        "vv::half2", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
          const double inv_m = 1.0 / masses(i);
          velocities(i,0) += 0.5*dt*forces(i,0)*inv_m;
          velocities(i,1) += 0.5*dt*forces(i,1)*inv_m;
          velocities(i,2) += 0.5*dt*forces(i,2)*inv_m;
        });
  }

  Kokkos::fence();
  write_pvd("particles.pvd", pvd_entries);
  std::cout << "\nWrote " << pvd_entries.size() << " VTP files + particles.pvd\n";
  std::cout << "Open particles.pvd in ParaView to view the time series.\n";
  std::cout << "Simulation complete.\n";
  return 0;
}
