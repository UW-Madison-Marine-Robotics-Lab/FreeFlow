# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jiayi Jin

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

import argparse
import json
from pathlib import Path
import math
import numpy as np
from utils import load_mesh_vertices, get_fin_ctrl_vertices, expand_action_range, eulerAnglesToRotationMatrix3D
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


# -------- argument parsing --------
parser = argparse.ArgumentParser(description="Render FreeFlow video")
parser.add_argument(
    '--cfg_path', type=str, 
    default='./task.json',
    help="Config for RL task")
parser.add_argument('--t_start', type=float, default=0.0, help="Start time")
parser.add_argument('--t_end', type=float, default=10.0, help="End time")
parser.add_argument('--plot_tip', type=bool, default=False, help="Plot fin tip trajectories")
args, _ = parser.parse_known_args()
cfg_path = args.cfg_path
t_start = args.t_start
t_end = args.t_end
plot_tip = args.plot_tip


# -------- read config --------
cfg = json.load(open(cfg_path, 'r'))
model_name = cfg["model_name"]
experiment_name = cfg["experiment_name"]
dim = cfg["dim"]
out_dir = Path(__file__).parent.parent / "output" / \
        experiment_name / model_name
out_dir = str(out_dir)

ray_num = int(cfg['ray_num'])
cnum = ray_num + 4
action_size = (3 if dim == 2 else 6 if dim == 3 else None) * cnum

ctrl_dt = cfg["interval"]

# read mesh and get control points
mesh_path = str(Path(__file__).parent.parent /
                        "assets" / "mesh" / f"{model_name}.mesh")
verts = load_mesh_vertices(mesh_path)
ctrl_idx = get_fin_ctrl_vertices(verts, ray_num)
ctrl_pos0 = verts[ctrl_idx]


# control range
ranges = cfg.get("action_range", {
            "translate": 0.2,
            "rotate": 0.2,
        })
n_trans = dim 
n_rot = 3 if dim == 3 else 1
trans_range = expand_action_range(ranges["translate"], n_trans)
rot_range = expand_action_range(ranges["rotate"], n_rot)

action_range = np.zeros(action_size)

for i in range(action_size):
    j = i % (n_trans + n_rot)
    if j < dim:
        action_range[i] = trans_range[j]
    else:
        action_range[i] = rot_range[j - n_trans]
action_range[-2 * (n_trans + n_rot):] = 0

# -------- read control sequence -------- 
action_rec = np.load(out_dir + "/action_record.npy")

step_start = math.floor(t_start / ctrl_dt)
step_end = math.floor(t_end / ctrl_dt)

shift = np.zeros((cnum, dim))
rotation = np.zeros(
    (cnum, 3, 3)) if dim == 3 else np.zeros((cnum))
ctrl_pos = np.zeros_like(ctrl_pos0)
ctrl_pos_history = []
for step in range(step_start, step_end):
    # process action
    actions = action_rec[step] * action_range
    for i in range(cnum):
        if dim == 2:
            shift[i] = actions[3*i:3*i+2]
            rotation[i] = actions[3*i+2]
            Rmat = np.asarray([
                [np.cos(rotation[i]), -np.sin(rotation[i])],
                [np.sin(rotation[i]),  np.cos(rotation[i])]
            ])
            ctrl_pos[i] = ctrl_pos0[i] @ Rmat.T + shift[i]
        else:
            shift[i] = actions[6*i:6*i+3]
            rotation[i] = eulerAnglesToRotationMatrix3D(actions[6*i+3:6*(i+1)])
            ctrl_pos[i] = ctrl_pos0[i] @ rotation[i].T + shift[i]
    
    ctrl_pos_history.append(ctrl_pos.copy())
    
    # plot fin tip
    if plot_tip:
        yz = ctrl_pos[ray_num: 2 * ray_num, 1:3]
        y = yz[:, 0]
        z = yz[:, 1]
        tck, u = splprep([y, z], s=0.0, k=min(3, len(y)-1))  # s=0 => interpolate points
        u_fine = np.linspace(0, 1, 100)
        y_s, z_s = splev(u_fine, tck)

        plt.plot(y_s, z_s, linewidth=1, alpha=0.35, label=f"step {step}")
        plt.scatter(y, z, s=8, alpha=0.35)
        # plt.plot(y, z, marker="o", linewidth=1, markersize=2, alpha=0.35, label=f"step {step}")


if plot_tip:
    plt.legend(fontsize=8, loc="best",
        ncol=2,          # helps when many steps
        frameon=True
    )
    plt.axis("equal")  # optional, helps interpret geometry
    plt.tight_layout()
    plt.show()

time = np.arange(len(ctrl_pos_history)) * ctrl_dt
u = time / time[-1]   # normalize parameter to [0, 1]
u_fine = np.linspace(0, 1, 5 * len(time))


# plot control inputs
def plot_spline_component(ax, i_range, comp_idx, title, ylabel):
    for i in i_range:
        traj = np.array([ctrl_pos_history[t][i, comp_idx] for t in range(len(ctrl_pos_history))])

        k = min(3, len(time) - 1)  # must be < #points
        tck, _ = splprep([traj], u=u, s=0.0, k=k)
        traj_smooth = splev(u_fine, tck)[0]

        ax.plot(time_smooth, traj_smooth, linewidth=1.5, alpha=0.5, label=f"cp {i}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.grid(True, alpha=0.25)

time_smooth = np.interp(u_fine, u, time) + t_start  # reuse for all curves
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

def plot_spline_component(ax, i_range, comp_idx, title, ylabel):
    for i in i_range:
        traj = np.array([ctrl_pos_history[t][i, comp_idx] for t in range(len(ctrl_pos_history))])

        k = min(3, len(time) - 1)  # must be < #points
        tck, _ = splprep([traj], u=u, s=0.0, k=k)
        traj_smooth = splev(u_fine, tck)[0]

        ax.plot(time_smooth, traj_smooth, linewidth=1.5, alpha=0.5, label=f"cp {i}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


# Row 1: i in range(ray_num)
plot_spline_component(
    axs[0, 0],
    range(ray_num),
    comp_idx=1,  # y
    title="Penducle y(t)",
    ylabel="y"
)
plot_spline_component(
    axs[0, 1],
    range(ray_num),
    comp_idx=2,  # z
    title="Penducle z(t)",
    ylabel="z"
)

# Row 2: i in range(ray_num, ray_num + 2)
plot_spline_component(
    axs[1, 0],
    range(ray_num, ray_num + 2),
    comp_idx=1,  # y
    title="Fin tip y(t)",
    ylabel="y"
)
plot_spline_component(
    axs[1, 1],
    range(ray_num, ray_num + 2),
    comp_idx=2,  # z
    title="Fin tip z(t)",
    ylabel="z"
)

for ax in axs[1, :]:
    ax.set_xlabel("time (s)")

# Legends: per-axes (can be crowded). If too crowded, comment these out.
for ax in axs.ravel():
    ax.legend(fontsize=8, ncol=1, frameon=True, loc="best")

plt.tight_layout()
plt.show()
