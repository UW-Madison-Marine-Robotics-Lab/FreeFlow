# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jiayi Jin

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

import argparse
import json
from pathlib import Path
from env import LBSEnv
import numpy as np
from utils import expand_action_range


# -------- argument parsing --------
parser = argparse.ArgumentParser(description="Render FreeFlow video")
parser.add_argument(
    '--cfg_path', type=str, 
    default='./task.json',
    help="Config for RL task")
args, _ = parser.parse_known_args()
cfg_path = args.cfg_path


# -------- read config --------
cfg = json.load(open(cfg_path, 'r'))
model_name = cfg["model_name"]
experiment_name = cfg["experiment_name"]
out_dir = Path(__file__).parent.parent / "output" / \
        experiment_name / model_name

ray_num = cfg['ray_num']
dim = cfg['dim']
action_size = (3 if dim == 2 else 6 if dim == 3 else None) * (ray_num + 2)
# cnum = ray_num + 4

total_time = cfg['total_time']
interval = cfg['interval']
max_steps = int(total_time / interval)

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

# load env
env = LBSEnv(cfg_path)
env.simulator.enableFluidSolver(False)  # turn off fluid
# -------- read control sequence -------- 
action_rec = np.load(out_dir + "/action_record.npy")

for step in range(max_steps):
    action = action_rec[step] 
    _, _, done, _ = env.step(action)

    if done:
        break
    
    env.render()
