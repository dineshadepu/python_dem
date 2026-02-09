import numpy as np
import trimesh
import matplotlib.pyplot as plt
from dataclasses import dataclass


def dot(a, b):
    return np.dot(a, b)


def cross(a, b):
    return np.cross(a, b)


def norm(v):
    return np.linalg.norm(v)


def normalize(v):
    n = norm(v)
    if n == 0:
        return v
    return v / n


def snap_to_face(A, B, C, P):
    # Returns (on_edge, closest_point)

    AB = B - A
    AC = C - A
    AP = P - A

    d1 = dot(AB, AP)
    d2 = dot(AC, AP)
    if d1 <= 0.0 and d2 <= 0.0:
        return True, A  # vertex A

    BP = P - B
    d3 = dot(AB, BP)
    d4 = dot(AC, BP)
    if d3 >= 0.0 and d4 <= d3:
        return True, B  # vertex B

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return True, A + v * AB  # edge AB

    CP = P - C
    d5 = dot(AB, CP)
    d6 = dot(AC, CP)
    if d6 >= 0.0 and d5 <= d6:
        return True, C  # vertex C

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return True, A + w * AC  # edge AC

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return True, B + w * (C - B)  # edge BC

    # Inside face
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return False, A + AB * v + AC * w


def triangle_sphere_CD(A, B, C, sphere_pos, radius):
    face_n = normalize(cross(B - A, C - A))
    h = dot(sphere_pos - A, face_n)

    on_edge, faceLoc = snap_to_face(A, B, C, sphere_pos)

    if not on_edge:
        depth = h - radius
        normal = face_n
        if depth >= 0.0:
            return False, None, None, None
        return True, normal, depth, faceLoc
    else:
        normal_d = sphere_pos - faceLoc
        dist = norm(normal_d)
        depth = dist - radius
        if dist == 0:
            return False, None, None, None
        normal = normal_d / dist
        if depth >= 0.0:
            return False, None, None, None
        return True, normal, depth, faceLoc


def create_plane(z=0.0, size=1.0):
    A = np.array([-size, -size, z])
    B = np.array([ size, -size, z])
    C = np.array([ size,  size, z])
    D = np.array([-size,  size, z])

    return [
        (A, B, C),
        (A, C, D)
    ]


class Particle():
    def __init__(self, x, y, z, radius, rho):
        self.x = x
        self.y = y
        self.z = z
        self.u = 0.
        self.v = 0.
        self.w = 0.
        # angular velocity
        self.wx = 0.
        self.wy = 0.
        self.wz = 0.

        self.overlap = 0.
        self.fn = 0.
        self.fx = 0.
        self.fy = 0.
        self.fz = 0.
        self.tor_x = 0.
        self.tor_y = 0.
        self.tor_z = 0.

        self.rho = rho
        self.radius = radius
        self.m = 4. / 3. * np.pi * radius**3. * self.rho
        # moment of inertia
        self.moi = 2. / 5. * self.m * radius**2.
        self.nu = 0.
        self.E = 0.
        self.G = 0.


class Wall():
    def __init__(self, x, y, normal_x, normal_y):
        self.x = x
        self.y = y
        self.normal_x = normal_x
        self.normal_y = normal_y


def update_velocity(particle, dt):
    particle.u += particle.fx/particle.m * dt
    particle.v += particle.fy/particle.m * dt
    particle.w += particle.fz/particle.m * dt
    particle.wx += particle.tor_x/particle.moi * dt
    particle.wy += particle.tor_y/particle.moi * dt
    particle.wz += particle.tor_z/particle.moi * dt


def update_position(particle, dt):
    particle.x += particle.u * dt
    particle.y += particle.v * dt
    particle.z += particle.w * dt

def make_forces_zero(particle):
    # compute the force
    particle.fx = 0.
    particle.fy = 0.
    particle.fz = 0.
    particle.tor_x = 0.
    particle.tor_y = 0.
    particle.tor_z = 0.

def body_force(particle, gx, gy, gz):
    # compute the force
    particle.fx += particle.m * gx
    particle.fy += particle.m * gy
    particle.fz += particle.m * gz


def distance(particle_1, particle_2):
    # compute the overlap
    xij = particle_1.x - particle_2.x
    yij = particle_1.y - particle_2.y
    zij = particle_1.z - particle_2.z
    return (xij**2. + yij**2. + zij**2.)**0.5


def compute_force_on_particles_due_to_particles(particle_1, particle_2, cor, fric_coeff, dt):
    # compute the overlap
    xij = particle_1.x - particle_2.x
    yij = particle_1.y - particle_2.y
    zij = particle_1.z - particle_2.z
    dist = distance(particle_1, particle_2)

    nij_x = xij / dist
    nij_y = yij / dist
    nij_z = zij / dist

    uij = particle_1.u - particle_2.u
    vij = particle_1.v - particle_2.v
    wij = particle_1.w - particle_2.w

    vn = uij * nij_x + vij * nij_y + wij * nij_z
    vn_x = vn * nij_x
    vn_y = vn * nij_y
    vn_z = vn * nij_z

    overlap = particle_1.radius + particle_2.radius - dist

    if overlap > 0:
        ############################
        # normal force computation #
        ############################
        # Compute stiffness
        # effective Young's modulus
        tmp_1 = (1. - particle_1.nu**2.) / particle_1.E
        tmp_2 = (1. - particle_2.nu**2.) / particle_2.E
        E_eff = 1. / (tmp_1 + tmp_2)
        tmp_1 = 1. / particle_1.radius
        tmp_2 = 1. / particle_2.radius
        R_eff = 1. / (tmp_1 + tmp_2)
        # Eq 4 [1]
        kn = 4. / 3. * E_eff * R_eff**0.5

        # compute damping coefficient
        tmp_1 = log(cor)
        tmp_2 = log(cor)**2. + pi**2.
        alpha_1 = -tmp_1 * (5. / tmp_2)**0.5
        tmp_1 = 1. / particle_1.m
        tmp_2 = 1. / particle_2.m
        m_eff = 1. / (tmp_1 + tmp_2)
        eta = alpha_1 * (m_eff * kn)**0.5 * overlap**0.25

        fn = kn * overlap**1.5
        fn_x = fn * nij_x - eta * vn_x
        fn_y = fn * nij_y - eta * vn_y
        fn_z = fn * nij_z - eta * vn_z

        particle_1.fn = fn
        particle_1.overlap = overlap
        particle_1.fx += fn_x
        particle_1.fy += fn_y
        particle_1.fz += fn_z

    else:
        fn = 0.
        particle_1.fn = 0.
        particle_1.fx = 0.
        particle_1.fy = 0.
        particle_1.fy = 0.


def get_shear_modulus(E, nu):
    return E / (2. * (1. + nu))


def run():
    # create the particle
    E = 4.8 * 1e10
    nu = 0.2
    G = get_shear_modulus(E, nu)
    fric_coeff = 0.350
    cor = 1.
    density = 2800
    radius = 0.01
    velocity = 10
    particle_1 = Particle(-radius - radius/10000, 0.0, 0.0, radius, density)
    particle_2 = Particle(radius, 0.0, 0.0, radius, density)
    particle_1.E = E
    particle_1.nu = nu
    particle_1.G = G
    particle_2.E = E
    particle_2.nu = nu
    particle_2.G = G
    particle_1.u = velocity
    particle_2.u = -velocity
    dt = 1e-7
    _t = 0
    tf = 80 * 1e-6
    gx = 0.
    gy = 0.
    gz = 0.
    t = []
    u = []
    y = []
    v = []
    fn = []
    overlap = []
    fy = []
    fx = []
    wz = []
    while _t < tf:
        y.append(particle_1.y)
        u.append(particle_1.u)
        v.append(particle_1.v)
        fn.append(particle_1.fn)
        overlap.append(particle_1.overlap)
        fy.append(particle_1.fy)
        fx.append(particle_1.fx)
        wz.append(particle_1.wz)
        t.append(_t)

        update_velocity(particle_1, dt)
        update_velocity(particle_2, dt)
        make_forces_zero(particle_1)
        make_forces_zero(particle_2)
        compute_force_on_particles_due_to_particles(particle_1, particle_2, cor, fric_coeff, dt)
        compute_force_on_particles_due_to_particles(particle_2, particle_1, cor, fric_coeff, dt)
        body_force(particle_1, gx, gy, gz)
        body_force(particle_2, gx, gy, gz)
        update_position(particle_1, dt)
        update_position(particle_2, dt)
        # print(_t)
        _t += dt

    # convert list into numpy arrays
    t = np.asarray(t) * 1000000
    overlap = np.asarray(overlap) * 1000000
    fn = np.asarray(fn) / 1000

    # plt.plot(t, y)
    # plt.savefig("y.jpg")
    # plt.clf()
    # plt.plot(t, v)
    # plt.savefig("v.jpg")
    # plt.clf()
    # plt.plot(t, u)
    # plt.savefig("u.jpg")
    # plt.clf()
    # plt.plot(t, fy)
    # plt.savefig("fy.jpg")
    # plt.clf()
    # plt.plot(t, wz)
    # plt.savefig("wz.jpg")
    # plt.clf()
    # plt.plot(t, fx)
    # plt.savefig("fx.jpg")

    # plt.clf()
    # plt.plot(t, fn)
    # plt.savefig("fn_t.jpg")

    plt.clf()
    path = os.path.abspath(__file__)
    # print(path)
    directory = os.path.dirname(path)
    data_fn_overlap_analytical = np.loadtxt(
        os.path.join(
            directory,
            'elastic_normal_impact_of_two_identical_spheres_analytical_fn_vs_overlap.csv'),
        delimiter=','
    )
    overlap_analytical, fn_analytical= data_fn_overlap_analytical[:, 0], data_fn_overlap_analytical[:, 1]
    plt.scatter(overlap_analytical, fn_analytical, label='Analytical')
    plt.plot(overlap, fn, "-", color="red", label='DEM')
    plt.xlabel(r'Normal contact displacement ($\mu$m)')
    plt.ylabel('Normal contact force (KN)')
    plt.legend()
    fig = os.path.join(directory, "fn_overlap.pdf")
    print(fig)
    plt.savefig(fig, dpi=300)


run()
