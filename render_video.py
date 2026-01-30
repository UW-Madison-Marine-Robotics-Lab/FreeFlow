# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jiayi Jin

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

from paraview.simple import *
from paraview import servermanager
import glob, re
import argparse


def force_white_background(view):
    # Disable palette / environment overrides
    view.UseColorPaletteForBackground = 0
    view.UseEnvironmentLighting = 0
    view.UseGradientEnvironmentalBG = 0
    view.UseTexturedEnvironmentalBG = 0

    # Force single-color background
    view.BackgroundColorMode = "Single Color"
    view.Background  = [1, 1, 1]
    view.Background2 = [1, 1, 1]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def series(pattern):
    files = sorted(glob.glob(pattern), key=natural_key)
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")
    return files


def union_bounds(b0, b):
    if b0 is None: return list(b)
    return [min(b0[0],b[0]), max(b0[1],b[1]),
            min(b0[2],b[2]), max(b0[3],b[3]),
            min(b0[4],b[4]), max(b0[5],b[5])]


def center(b):
    return [(b[0]+b[1])*0.5, (b[2]+b[3])*0.5, (b[4]+b[5])*0.5]


# -------- argument parsing --------
parser = argparse.ArgumentParser(description="Render FreeFlow video")
parser.add_argument(
    "--job-name",
    required=True,
    help="Job folder name under ./output/, e.g. fish3d_lbs"
)
parser.add_argument(
    "--view",
    choices=["top", "back", "side"],
    default="top",
    help="Camera view direction: top (default), back, or side"
)
args, _ = parser.parse_known_args()

job = args.job_name
view_mode = args.view

# -------- inputs --------
fluid_files = series(f"./output/{job}/fluid_frame_*.vtk")  # XML ImageData, mislabeled .vtk
solid_files = series(f"./output/{job}/solid_frame_*.vtk")  # XML UnstructuredGrid, mislabeled .vtk

fluid = XMLImageDataReader(FileName=fluid_files)
solid = XMLUnstructuredGridReader(FileName=solid_files)

# -------- time --------
scene = GetAnimationScene()
scene.PlayMode = "Snap To TimeSteps"
scene.UpdateAnimationUsingDataTimeSteps()
tk = scene.TimeKeeper
times = list(tk.TimestepValues)

# -------- vorticity + Q from velocity --------
grad = Gradient(Input=fluid)
grad.ScalarArray = ["POINTS", "velocity"]
grad.ComputeVorticity = 1
grad.ComputeQCriterion = 1  # produces "Q Criterion" in your build

# vorticity magnitude for coloring
omega = Calculator(Input=grad)
omega.ResultArrayName = "omega_mag"
omega.Function = "mag(Vorticity)"

# -------- contour Q Criterion (better vortex rings) --------
cont = Contour(Input=omega)
cont.ContourBy = ["POINTS", "Q Criterion"]   # <-- exact name from your output

# Tune these isovalues to get more/less rings:
# If you see nothing -> divide by 10
# If it's a big blob -> multiply by 10
cont.Isosurfaces = [0.005, 0.01, 0.02, 0.05, 0.1]

surf = ExtractSurface(Input=cont)

# -------- view --------
view = GetActiveViewOrCreate("RenderView")

force_white_background(view)

view.OrientationAxesVisibility = 0

# show fish
sd = Show(solid, view)
sd.Representation = "Surface"
sd.Opacity = 1.0

# show wake
wd = Show(surf, view)
wd.Representation = "Surface"
wd.Opacity = 0.1
ColorBy(wd, ("POINTS", "omega_mag"))
wd.RescaleTransferFunctionToDataRange(True, False)

# transparency quality (optional)
try:
    view.UseDepthPeeling = 1
    view.MaximumNumberOfPeels = 200
    view.OcclusionRatio = 0.0
except Exception:
    pass

# -------- fixed camera: fit full fish trajectory --------
traj = None
for t in times:
    tk.Time = t
    solid.UpdatePipeline(t)
    traj = union_bounds(traj, solid.GetDataInformation().GetBounds())

c = center(traj)
dx, dy, dz = traj[1]-traj[0], traj[3]-traj[2], traj[5]-traj[4]
span = max(dx, dy, dz)

view.CameraParallelProjection = 1
view.CameraFocalPoint = c

if view_mode == "top":
    # top-down camera (looking along -Z)
    view.CameraFocalPoint = c
    view.CameraPosition   = [c[0], c[1], c[2] + 3.0*span]   # higher = more zoomed out
    view.CameraViewUp     = [0, 1, 0]                       # keep +Y as "up" on screen
    # zoom: use X/Y span since we look down Z
    view.CameraParallelScale = 0.55 * max(dx, dy)           # increase if fish leaves frame
elif view_mode == "back":
    # back view 
    view.CameraPosition   = [c[0] - 3.0*span, c[1], c[2]]
    view.CameraViewUp     = [0, 0, 1]
    view.CameraParallelScale = 0.55 * max(dy, dz)
elif view_mode == "side":
    # side view
    view.CameraPosition = [c[0], c[1] - 3.0 * span, c[2]]
    view.CameraViewUp   = [0, 0, 1]
    # zoom: use X/Z span
    view.CameraParallelScale = 0.55 * max(dx, dz)


# -------- save AVI --------
SaveAnimation(f"./output/{job}/video_{view_mode}.avi", view, FrameRate=10, ImageResolution=[1920, 1080])
print(f"Saved video_{view_mode}.avi")
