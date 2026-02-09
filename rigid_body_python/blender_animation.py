import bpy
import numpy as np
import os

CSV_PATH = "/Users/dineshadepu/postdoc_dwi/softwares/rigid_body_python"
OBJ_NAME = "Particles"   # keep Plane since it works

def update_particles(scene):
    frame = scene.frame_current
    fname = os.path.join(CSV_PATH, f"t_body_{frame:04d}.csv")

    if not os.path.exists(fname):
        return

    pts = np.loadtxt(fname, delimiter=",")

    obj = bpy.data.objects[OBJ_NAME]
    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(pts.tolist(), [], [])
    mesh.update()

# Remove old handlers safely
bpy.app.handlers.frame_change_pre[:] = [
    h for h in bpy.app.handlers.frame_change_pre
    if h.__name__ != "update_particles"
]

bpy.app.handlers.frame_change_pre.append(update_particles)
print("CSV animation system installed.")

Emission material
3 lights
Motion trails
