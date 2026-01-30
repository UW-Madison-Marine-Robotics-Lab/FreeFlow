# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jiayi Jin

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import taichi as ti
from pathlib import Path
import sys
import os
import numpy as np

import fsi_simulator as fsi

NX = 512
NY = 128
NZ = 128

mesh_path = Path(__file__).parent.parent / "assets" / "mesh" / "bluegill_sunfish.mesh"

# Fin area geometry
PENDUCLE_Z_LB = -45e-3
PENDUCLE_Z_UB = 45e-3
FIN_POST_LB = (-174e-3, -127e-3)      # (X, Z)
FIN_POST_UB = (-174e-3, 127e-3)       # (X, Z)
RAY_NUM = 6

def load_mesh_vertices(mesh_path):
    """
    read vertices from .mesh file
    """
    with open(mesh_path, "r") as f:
        lines = f.readlines()

    # find "Vertices"
    i = 0
    while i < len(lines) and lines[i].strip() != "Vertices":
        i += 1
    if i >= len(lines):
        raise RuntimeError(f"No 'Vertices' section found in {mesh_path}")

    n = int(lines[i + 1].strip())
    verts = np.zeros((n, 3), dtype=np.float64)
    for k in range(n):
        parts = lines[i + 2 + k].split()
        # parts: [x, y, z]
        verts[k, 0] = float(parts[0])
        verts[k, 1] = float(parts[1])
        verts[k, 2] = float(parts[2])

    return verts


def find_closest_vertex(verts, query, wy=10.0):
    """
    Find index of closest vertex to a given point.
    
    :param verts: Vertex positions
    :param query: Query point [x, y, z]
    :param wy: weight on y-direction (>=1). Larger -> stronger centering.
    """
    query = np.asarray(query, dtype=np.float64)

    dx = verts[:, 0] - query[0]
    dy = verts[:, 1] - query[1]
    dz = verts[:, 2] - query[2]

    # anisotropic metric
    dist2 = dx*dx + wy * dy*dy + dz*dz

    idx = np.argmin(dist2)
    return idx, np.sqrt(dist2[idx])


def find_min_x_at_z(verts, z0, z_tol=5e-3, y_weight=0.0):
    """
    Find vertex index with smallest x among vertices
    whose z-coordinate is close to z0.
    
    :param verts: Vertex positions
    :param z0: Target z value
    :param z_tol: Allowed tolerance in z
    :param y_weight: Choose the one closest to the center plane y=0.
        If y_weight > 0, it softly penalizes |y|.
    """
    z = verts[:, 2]
    mask = np.abs(z - z0) <= z_tol
    if not np.any(mask):
        raise ValueError(f"No vertices found with |z - {z0}| <= {z_tol}")

    idxs = np.where(mask)[0]
    x_vals = verts[idxs, 0]
    y_vals = np.abs(verts[idxs, 1])

    # Two options:
    # (A) hard: choose smallest x, break ties by smallest |y|
    # (B) soft: minimize x + y_weight*|y|
    if y_weight == 0.0:
        # lexsort: primary x, secondary |y|
        order = np.lexsort((y_vals, x_vals))
        idx = idxs[order[0]]
    else:
        score = x_vals + y_weight * y_vals
        idx = idxs[np.argmin(score)]

    return idx, verts[idx]


def get_fin_ctrl_vertices(verts, N=6):
    """
    Get fin control vertices, size 2N+1
    
    :param verts: Vertex positions
    :param int: num of rays
    """
    ctrl_idx = np.empty(N * 2 + 1)

    z_penducle = np.linspace(PENDUCLE_Z_LB, PENDUCLE_Z_UB, N)    # penducle height
    z_fin = np.linspace(FIN_POST_LB[1], FIN_POST_UB[1], N)       # fin posterior height
    for i in range(N):
        ctrl_idx[i], _ = find_closest_vertex(verts, [0, 0, z_penducle[i]])
        ctrl_idx[N + i], _ = find_min_x_at_z(verts, z_fin[i])
    
    # frontmost vertex
    ctrl_idx[-1] = np.argmax(verts[:, 0])

    return ctrl_idx.astype(int)


def find_vertices_on_xz_segment(verts, p0, p1, dist_tol):
    """
    Find vertices on a xz plane segment
    
    :param verts: Vertex positions
    :param p0: peduncle start point (x0, z0)
    :param p1: fin posterior end point (x1, z1)
    :param dist_tol: Max allowed perpendicular distance (in x-z plane)
    """

    p0 = np.asarray(p0, dtype=np.float64)  # (2,)
    p1 = np.asarray(p1, dtype=np.float64)  # (2,)

    xz = verts[:, [0, 2]].astype(np.float64)  # (N,2)
    v = p1 - p0
    vv = np.dot(v, v)
    if vv == 0.0:
        # Degenerate segment: treat as "near a point"
        d2 = np.sum((xz - p0[None, :])**2, axis=1)
        return np.where(d2 <= dist_tol**2)[0]

    w = xz - p0[None, :]                  # (N,2)
    t = (w @ v) / vv                      # projection scalar, (N,)
    t_clamped = np.clip(t, 0.0, 1.0)

    proj = p0[None, :] + t_clamped[:, None] * v[None, :]   # (N,2)
    d2 = np.sum((xz - proj)**2, axis=1)    # squared distance to segment in x-z

    # Require: close to segment AND projection falls within segment (not just endpoints due to clamp)
    on_segment = (t >= 0.0) & (t <= 1.0) & (d2 <= dist_tol**2)

    return np.where(on_segment)[0]


def get_fin_ray_regions(verts, ctrl_idx, dist_tol=1e-2):
    """
    Get the N fin ray regions, return a list of ray regions.
    
    :param verts: Vertex positions
    :param ctrl_idx: fin control vertices
    :param dist_tol: Max allowed perpendicular distance to ray segment 
        (half the width)
    """

    N = (len(ctrl_idx) - 1) // 2
    ray_regions = [np.empty(0) for _ in range(N)]

    for i in range(N):
        p0 = (verts[ctrl_idx[i], 0], verts[ctrl_idx[i], 2])
        p1 = (verts[ctrl_idx[i + N], 0], verts[ctrl_idx[i + N], 2])
        ray_regions[i] = find_vertices_on_xz_segment(verts, p0, p1, dist_tol)
    
    return ray_regions


def make_config(config_path: str, output_path: str):
    config = {
        "dimension": 3,
        "fluid_viscosity": 0.02,
        "fluid_density": 500.0,
        "fluid_nx": NX,
        "fluid_ny": NY,
        "fluid_nz": NZ,
        "fluid_dx": 0.02,
        "solid_solver_type": "vbd",
        "total_time": 10.0,
        "dt": 5e-3,
        "output_frequency": 200,
        "output_path": output_path,
        "log_level": "info",
        "log_file": "simulation_3d.log",
        "global_fem_options": {
            "optimizer_type": "newton",
            "iterations": 40,
            "verbose_level": 1,
            "line_search_method": "backtracking",
            "force_density_abs_tol": 1e-2,
            "ls_max_iter": 20,
            "ls_beta": 0.3,
            "ls_alpha": 1e-4,
            "linear_solver_type": "cholmod_ldlt",
            "grad_check": False,
            "substeps": 3,
            "vbd_iterations": 30,
            "omega": 0.8
        },
        "solids": [],
        "boundaries": [],
        "boundary_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    nx = config["fluid_nx"]
    ny = config["fluid_ny"]
    nz = config["fluid_nz"]
    dx = config["fluid_dx"]

    verts = load_mesh_vertices(mesh_path)

    # get control vertices
    ctrl_idx = get_fin_ctrl_vertices(verts, RAY_NUM)

    # assign stiffness per node
    ray_regions = get_fin_ray_regions(verts, ctrl_idx)
    ray_set = set(np.concatenate(ray_regions).astype(int))

    # plt.figure()
    # xz_ray = verts[np.array(list(ray_set), dtype=int)][:, [0, 2]]
    # xz_ctrl = verts[np.asarray(ctrl_idx, dtype=int)][:, [0, 2]]
    # plt.scatter(xz_ray[:, 0], xz_ray[:, 1], s=5, label="Ray vertices")
    # plt.scatter(
    #     xz_ctrl[:-1, 0], xz_ctrl[:-1, 1],
    #     s=40, c="red", marker="o", label="Control points"
    # )
    # plt.xlabel("x")
    # plt.ylabel("z")
    # plt.axis("equal")
    # plt.show()

    stiffness_per_node = [
        1e7 if i in ray_set else
        .5e6 if verts[i, 0] < 0.02 else
        1e6
        for i in range(len(verts))
    ]

    solid_body = {
        "mesh_path": str(mesh_path),
        "type": 'static',
        "density": 1000.0,
        # "youngs_modulus": 1e5,
        "youngs_modulus_per_node": stiffness_per_node,
        "poisson_ratio": 0.45,
        "translate": [0.25 * nx * dx, 0.5 * ny * dx, 0.5 * nz * dx],
        "scale": [1., 1., 1.],
        "lbs_control_config": {
            "cnum": RAY_NUM * 2 + 1,
            "omega": 0.3,
            "stiffness": 10.0,
            "ctrl_idx": ctrl_idx.tolist()
        }
    }
    config["solids"].append(solid_body)

    # print(config)

    boundaries = []

    for j in range(ny):
        for k in range(nz):
            boundaries.append({"type": "OutletLeft", "pos": [0, j, k]})
            boundaries.append({"type": "OutletRight", "pos": [nx - 1, j, k]})

    for i in range(nx):
        for k in range(nz):
            boundaries.append({"type": "OutletFront", "pos": [i, 0, k]})
            boundaries.append({"type": "OutletBack", "pos": [i, ny - 1, k]})
            # boundaries.append({"type": "Wall", "pos": [i, 0, k]})
            # boundaries.append({"type": "Wall", "pos": [i, ny - 1, k]})

    for i in range(nx):
        for j in range(ny):
            boundaries.append({"type": "OutletDown", "pos": [i, j, 0]})
            boundaries.append({"type": "OutletUp", "pos": [i, j, nz - 1]})
            # boundaries.append({"type": "Wall", "pos": [i, j, 0]})
            # boundaries.append({"type": "Wall", "pos": [i, j, nz - 1]})

    config["boundaries"] = boundaries

    import json
    try:
        output_dir = os.path.dirname(config_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
        return config
    except IOError as e:
        print(f"Error: Failed to write to file {config_path}. Reason: {e}")
        exit(-1)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


def run_test():
    config_path = Path(__file__).parent.parent / "assets" / \
        "configs" / "fish3d_fin.json"
    output_path = Path(__file__).parent.parent / "output" / "fish3d_fin"
    # if not config_path.exists():
    make_config(str(config_path), str(output_path))
    config_loader = fsi.Config3D()

    config_loader.load(str(config_path))
    params = config_loader.get_params()

    simulator = fsi.Simulator3D(params)
    simulator.initialize()

    gui = ti.GUI("LBM3D", (NX, NY), show_gui=True)

    def vis_slice(f, type="magnitude", slice_idx=50):
        fMom1 = simulator.get_fluid_moments().transpose(2, 1, 0, 3)
        if type == "magnitude":
            vel = (fMom1[:, :, slice_idx, 1] ** 2 + fMom1[:, :,
                   slice_idx, 2] ** 2 + fMom1[:, :, slice_idx, 3] ** 2) ** 0.5
            vel_img = cm.plasma(vel / 0.3)
        elif type == "vorticity":
            u = fMom1[:, :, slice_idx, 1]
            v = fMom1[:, :, slice_idx, 2]

            du_dy = np.gradient(u, axis=1)
            dv_dx = np.gradient(v, axis=0)

            vor = du_dy - dv_dx
            # vor[flag == lbm.SOLID_DYNAMIC] = 0.02
            colors = [
                (151/255, 139/255, 229/255),
                (255/255, 255/255, 255/255),
                (209/255, 83/255, 124/255),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "my_cmap", colors)
            vel_img = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)

        gui.set_image(vel_img)

        gui.show(str(output_path / f"{f:03d}.png"))

    simulator.begin_profiler("3d bluegill sunfish actuated by lbs control")
    axis = np.array([0.0, 0.0, 1.0])
    for i in range(2000):
        if i % 4 == 0:  
            pos = np.array(simulator.getVertices())
            x = pos.mean(axis=0)
            print(f"Step {i} mean x {x}")
            # angle = np.sin(2 * np.pi * i / 200) * np.pi / 4   # flat motion
            angles = [np.sin(2 * np.pi * (i / 200 + j / (RAY_NUM * 4))) * np.pi / 4 for j in range(RAY_NUM)]    # undulation
            heave = np.sin(2 * np.pi * i / 200) * 1e-2
            shift = np.zeros((RAY_NUM * 2 + 1, 3))
            shift[:RAY_NUM, 1] = heave
            rotation = np.array(
                [np.eye(3)] * RAY_NUM +
                [rotation_matrix(axis, -angles[j]) for j in range(RAY_NUM)] +
                [np.eye(3)]
            )
            simulator.apply_lbs_control(0, shift, rotation)
        simulator.step()
        if i % 20 == 0:
            # simulator.save_frame_data(i // 20, True, True)
            vis_slice(i // 20, "vorticity", NZ // 2)
    simulator.end_profiler()


if __name__ == "__main__":
    run_test()
