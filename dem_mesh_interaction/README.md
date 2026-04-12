# DEM Particle-Mesh Interaction

Particles fall under gravity into a triangulated 3-D open-top tank.
Contact forces (Hertz normal only) are computed **per triangle** using an
ArborX BVH.  The mesh BVH is built once (static mesh) and queried every
time step.

The tank has **five faces**: a floor at `z = 0` and four vertical walls at
`x = 0`, `x = L`, `y = 0`, `y = L`.  The top is open so particles can be
dropped in.  Because forces are computed as

```
n = (particle_centre − closest_point_on_triangle) / dist
```

the normal always points inward regardless of which face or edge is in
contact, with no need to store or compute face normals explicitly.

---

## Table of Contents

1. [Overview](#overview)
2. [Physics](#physics)
3. [ArborX Usage Pattern](#arborx-usage-pattern)
4. [Code Structure](#code-structure)
5. [Build and Run](#build-and-run)
6. [Expected Output](#expected-output)
7. [Extending the Example](#extending-the-example)

---

## Overview

The example demonstrates the core pattern for DEM particle-mesh contact in
ArborX:

| Step | What happens |
|------|-------------|
| Build BVH | Once, from the triangulated mesh (static geometry) |
| Each time step | Query BVH with one sphere per particle |
| Callback | For each (particle, triangle) hit, compute closest point → force |
| Integration | Velocity Verlet |

The scenario: 9 spherical particles (3×3 grid) start above a flat square
floor made of 128 triangles and fall under gravity.  Without damping the
particles bounce elastically.

---

## Physics

### Hertz Normal Contact Force

For a sphere of radius `R` pressing against a flat wall:

```
E_eff = 1 / ( (1 - nu_p^2)/E_p  +  (1 - nu_w^2)/E_w )

delta = R - dist(particle_centre, closest_point_on_triangle)

fn = (4/3) * E_eff * sqrt(R) * delta^(3/2)    [if delta > 0]

n  = (particle_centre - closest_point) / dist   [outward normal]

F  = fn * n
```

`R_eff = R_particle` because the wall has infinite radius.

Default material parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `E_p` | 1×10⁷ Pa | particle Young's modulus |
| `nu_p` | 0.3 | particle Poisson ratio |
| `E_w` | 1×10⁷ Pa | wall Young's modulus |
| `nu_w` | 0.3 | wall Poisson ratio |
| `rho` | 2500 kg/m³ | particle density |
| `gz` | −9.81 m/s² | gravity |

> **Note:** Only the normal (repulsive) force is implemented.  There is no
> tangential friction, rolling resistance, or normal damping.  The result is
> purely elastic bouncing.

### Integration: Velocity Verlet

```
v_{n+1/2}  =  v_n  +  0.5 * dt * f_n / m
x_{n+1}    =  x_n  +  dt  * v_{n+1/2}
f_{n+1}    =  gravity + contact_force( x_{n+1} )
v_{n+1}    =  v_{n+1/2}  +  0.5 * dt * f_{n+1} / m
```

---

## ArborX Usage Pattern

### 1 — Build the BVH from mesh triangles (once)

```cpp
using Triangle3d = ArborX::Triangle<3, double>;

// Floor + 4 walls.  fn = cells per unit length; fh scales with H/L.
Kokkos::View<Triangle3d*, MemorySpace> triangles = make_tank_triangles(L, H, fn);

ArborX::BoundingVolumeHierarchy bvh{
    exec_space, ArborX::Experimental::attach_indices(triangles)};
```

`attach_indices` wraps each triangle so that in the callback `val.index`
gives the triangle's position in the original view.

### 2 — Define sphere queries (one per particle)

```cpp
struct ParticleQueries {
  Kokkos::View<Point3d*, MemorySpace> positions;
  Kokkos::View<double*,  MemorySpace> radii;
};

template <>
struct ArborX::AccessTraits<ParticleQueries> {
  using memory_space = MemorySpace;
  static KOKKOS_FUNCTION std::size_t size(ParticleQueries const& q) {
    return q.positions.extent(0);
  }
  static KOKKOS_FUNCTION auto get(ParticleQueries const& q, std::size_t i) {
    return ArborX::intersects(Sphere3d{q.positions(i), q.radii(i) * 1.5});
  }
};
```

The search sphere radius is `1.5 * R` — a small buffer beyond the contact
distance so that triangles at the exact contact boundary are never missed.

### 3 — Force callback (2-argument form)

When calling `bvh.query()` **without** output views the callback signature is:

```cpp
template <typename Predicate, typename Value>
KOKKOS_FUNCTION void operator()(Predicate const& pred, Value const& val) const;
```

Inside the callback:

```cpp
int i = static_cast<int>(ArborX::getData(pred)); // particle index
int j = val.index;                                // triangle index

auto cp = ArborX::Experimental::closestPoint(positions(i), triangles(j));
// ... compute overlap and force ...
Kokkos::atomic_add(&forces(i, 0), fn * nx);
```

`ArborX::getData(pred)` returns the integer attached by `attach_indices<int>`
on the query side.  `val.index` returns the integer attached by
`attach_indices` on the tree side.

> **Important:** Use `Kokkos::atomic_add` because multiple triangles may be
> in contact with the same particle simultaneously.

### 4 — Query each time step

```cpp
bvh.query(exec_space,
          ArborX::Experimental::attach_indices<int>(
              ParticleQueries{positions, radii}),
          MeshContactCallback{positions, forces, triangles, radii, E_eff});
```

The BVH is **not** rebuilt each step because the mesh is static.  Only the
queries (sphere positions) change.

---

## Code Structure

```
example_dem.cpp
│
├── Type aliases
│     Point3d / Triangle3d / Sphere3d   (ArborX::Point/Triangle/Sphere<3, double>)
│
├── ParticleQueries          AccessTraits — sphere intersect query per particle
│
├── MeshContactCallback      2-arg callback — Hertz force per (particle, triangle)
│     closestPoint → overlap → fn → atomic_add to forces
│
├── make_tank_triangles()    Build floor + 4 walls via bilinear quad tessellation
│     add_face() lambda      Tessellates one rectangular face into 2·ncx·ncy triangles
│
└── main()
      ├── Parse CLI args (--tank-l, --tank-h, --mesh-n, ...)
      ├── Build BVH from all tank triangles (once — static geometry)
      ├── Write tank.vtp
      ├── Init particle state (positions inside tank, staggered layers)
      ├── Init forces (gravity)
      └── Velocity-Verlet time loop
            half1 → pos → gravity_reset → bvh.query → half2 → [output]
```

### Key types

| Type | Description |
|------|-------------|
| `Point3d` | `ArborX::Point<3, double>` — particle centre |
| `Triangle3d` | `ArborX::Triangle<3, double>` — mesh triangle |
| `Sphere3d` | `ArborX::Sphere<3, double>` — search sphere |
| `ParticleQueries` | Custom struct with `AccessTraits` specialisation |
| `MeshContactCallback` | Functor computing Hertz normal force per hit pair |

### Key views

| View | Layout | Contents |
|------|--------|----------|
| `positions` | `Point3d*` | particle centres |
| `velocities` | `double*[3]` | particle velocities |
| `forces` | `double*[3]` | accumulated forces (reset each step) |
| `radii` | `double*` | particle radii |
| `masses` | `double*` | particle masses |
| `triangles` | `Triangle3d*` | mesh triangles (static) |

---

## Build and Run

The example is built as part of the ArborX examples:

```bash
cd <arborx_build_dir>
make ArborX_Example_DEM_Mesh_Interaction.exe -j4
```

Run with defaults (1024 particles, 60 000 steps):

```bash
./examples/dem_mesh_interaction/ArborX_Example_DEM_Mesh_Interaction.exe
```

Available options:

| Flag | Default | Description |
|------|---------|-------------|
| `--nx N` | 16 | particles per row in x |
| `--ny N` | 16 | particles per row in y |
| `--layers N` | 4 | particle layers stacked in z |
| `--radius r` | 0.01 | particle radius [m] |
| `--dt dt` | 1e-5 | time step [s] |
| `--steps N` | 60000 | number of time steps |
| `--out-every N` | 1000 | output interval (steps) |
| `--tank-l L` | 0.5 | tank side length [m] |
| `--tank-h H` | 0.5 | tank wall height [m] |
| `--mesh-n N` | 16 | mesh cells per unit length |

Example — tall narrow tank with more particles:

```bash
./ArborX_Example_DEM_Mesh_Interaction.exe \
    --nx 12 --ny 12 --layers 8 \
    --tank-l 0.3 --tank-h 0.8 \
    --steps 80000
```

---

## Output Files

All files are written to the current working directory (the build directory
when running via `make`).

| File | Written | Contents |
|------|---------|----------|
| `tank.vtp` | Once | Triangulated tank (floor + 4 walls) |
| `particles_NNNNNN.vtp` | Each output step | Particle positions, velocities, radii |
| `particles.pvd` | End of run | ParaView time-series descriptor |

---

## Visualising in ParaView

### Load the time series

1. Open ParaView.
2. **File → Open** → select `particles.pvd`.  ParaView automatically loads
   all `particles_NNNNNN.vtp` files and creates a time slider.
3. In the same session, **File → Open** → select `tank.vtp` to overlay the
   tank geometry.
4. Click **Apply** for each loaded source.
5. To see inside the tank, enable **Wireframe** rendering for `tank.vtp`
   (toolbar → Surface → Wireframe) or reduce its **Opacity** to ~0.3.

### Render particles as spheres

The VTP files store particles as point vertices.  To render them as spheres:

1. Select the `particles.pvd` source in the Pipeline Browser.
2. **Filters → Alphabetical → Glyph** (or search "Glyph").
3. In the Glyph properties panel:
   - **Glyph Type**: `Sphere`
   - **Scale Array**: `radius`
   - **Scale Factor**: `2.0`  (diameter = 2 × radius)
   - **Orientation Array**: `No orientation array`
4. Click **Apply**.

### Colour by velocity magnitude

1. Select the Glyph source.
2. In the toolbar change the colouring drop-down from `Solid Color` to
   `velocity`.
3. Click the **Rescale** button to fit the colour range to the current frame.
4. Use **Filters → Common → Calculator** to add a `velocity_magnitude` array:
   `mag(velocity)` if you prefer a scalar colour map.

### Animate

- Use the **VCR controls** (play/pause/step) in the toolbar to step through
  time.
- **File → Save Animation** exports an image sequence or video.

---

## Expected Output

```
DEM Particle-Mesh Interaction (Hertz normal, no p-p contact)
  Backend:         OpenMP
  Particles:       1024  (16 x 16 x 4 layers)
  Radius:          0.01 m
  E_eff:           5.49451e+06 Pa
  dt:              1e-05 s
  Steps:           60000
  Output every:    1000 steps
  Tank:            0.5 x 0.5 x 0.5 m
  Tank triangles:  2560

Wrote tank.vtp
step, time [s], z_min [m], z_avg [m], z_max [m]
0,     0,     0.05,      0.0875,    0.125
2000,  0.02,  0.048038,  0.085538,  0.123038
4000,  0.04,  0.042152,  0.079652,  0.117152
...

Wrote 61 VTP files + particles.pvd
Open particles.pvd in ParaView to view the time series.
```

Interpretation:

- `z_min = 0.05` at step 0 — the lowest layer starts at z = 5R = 0.05 m.
- `z_max = 0.125` — the top layer starts at z = 5R + 3 × 2.5R.
- All z values decrease under gravity until the bottom layer contacts the floor
  (`z_min → R = 0.01`).
- Without damping the particles bounce elastically forever.

---

## Extending the Example

### Add normal damping

Inside `MeshContactCallback::operator()`, after computing `fn`, add a
viscous damping term using the relative normal velocity:

```cpp
// velocity of particle i along the contact normal
double v_rel_n = velocities(i, 0)*nx + velocities(i, 1)*ny + velocities(i, 2)*nz;

// damping coefficient from coefficient of restitution e:
//   beta  = -ln(e) / sqrt(pi^2 + ln(e)^2)
//   gamma = 2 * beta * sqrt(m * S_n)   where S_n = 2*E_eff*sqrt(R*delta)
double S_n      = 2.0 * E_eff * Kokkos::sqrt(r * overlap);
double beta     = /* from cor */;
double gamma    = 2.0 * beta * Kokkos::sqrt(masses(i) * S_n);
double fn_damp  = -gamma * v_rel_n;

double fn_total = fn + fn_damp;
// clamp to avoid tensile normal force
if (fn_total < 0.0) fn_total = 0.0;
```

The callback needs access to `velocities` and `masses` — add them as
additional member views.

### Add tangential (friction) force

1. Store a per-particle tangential displacement `View<double*[3]>` that
   persists between steps.
2. In the callback, compute the tangential relative velocity
   `v_t = v_rel - (v_rel·n)*n`, increment the displacement, compute a
   spring-dashpot tangential force, and apply the Coulomb limit
   `|F_t| ≤ mu * |F_n|`.
3. Reset displacements for particles with no contact (handle in a separate
   kernel after the query).

### Dynamic mesh (moving geometry)

If the mesh moves each step, rebuild the BVH inside the time loop:

```cpp
for (int step = 1; step <= steps; ++step) {
    update_mesh_positions(triangles, step * dt);   // your function

    ArborX::BoundingVolumeHierarchy bvh{
        exec_space, ArborX::Experimental::attach_indices(triangles)};

    // ... rest of the step ...
}
```

BVH construction is O(N log N) and is typically fast enough for moderately
sized meshes even when rebuilt every step.

### Load an STL/OBJ mesh

Replace `make_floor_triangles()` with a function that reads triangles from
a file and copies them into a `Kokkos::View<Triangle3d*, MemorySpace>`.  The
rest of the code is unchanged.
