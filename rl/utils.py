# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Changyu Hu

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

# --------------------------------------------------------------------------------
# Modifications Copyright 2026 Jiayi Jin
#
# This file has been significantly modified from its original version in
# the Popular-RL-Algorithms library. The original license and copyright
# notices are retained above.
#
# The modifications are provided under the terms of the license of this project.
# --------------------------------------------------------------------------------

from scipy.sparse import csr_array
import pathlib
import struct
import shutil
import numpy as np

np_integer, np_real = np.int32, np.float32

# Fin area geometry
FIN_ANT_LB = (-100e-3, -45e-3)
FIN_ANT_UB = (-100e-3, 45e-3)
FIN_POST_LB = (-274e-3, -127e-3)      # (X, Z)
FIN_POST_UB = (-274e-3, 127e-3)       # (X, Z)
PENDUCLE_LB = -95e-3
PENDUCLE_UB = 70e-3     # (X, Z)


def to_real_array(val):
    return np.array(val, dtype=np_real).copy()


def to_integer_array(val):
    return np.array(val, dtype=np_integer).copy()


def eulerAnglesToRotationMatrix3D(theta):
    """
    Convert euler angles to rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def create_folder(folder_name, exist_ok):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=exist_ok)


def delete_folder(folder_name):
    shutil.rmtree(folder_name)


def delete_file(file_name):
    pathlib.Path(file_name).unlink()


def file_exist(file_name):
    return pathlib.Path(file_name).is_file()


# Load Eigen matrices and vectors.


def load_real_vector(file_name):
    with open(file_name, "rb") as f:
        content = f.read()
        # The first 8 bytes are the vec size.
        num = np_integer(struct.unpack("=q", content[:8])[0])
        data = struct.unpack("={:d}d".format(num), content[8:])
        return to_real_array(data).ravel()


def sparse_matrix_to_triplets(mat):
    row_num, col_num = mat.shape
    row_num = np_integer(row_num)
    col_num = np_integer(col_num)
    nonzeros_num = mat.nnz
    nonzeros_num = np_integer(nonzeros_num)

    triplets = []
    for r in range(row_num):
        cols = mat.indices[mat.indptr[r]:mat.indptr[r + 1]]
        data = mat.data[mat.indptr[r]:mat.indptr[r + 1]]
        for c, v in zip(cols, data):
            # Triplet (r, c, v).
            triplets.append((np_integer(r), np_integer(c), np_real(v)))
    return triplets


def triplets_to_sparse_matrix(row_num, col_num, triplets):
    row_num = np_integer(row_num)
    col_num = np_integer(col_num)
    row_idx = []
    col_idx = []
    data = []
    for r, c, v in triplets:
        row_idx.append(r)
        col_idx.append(c)
        data.append(v)
    row_idx = to_integer_array(row_idx)
    col_idx = to_integer_array(col_idx)
    data = to_real_array(data)
    # CSR matrix format.
    return csr_array((data, (row_idx, col_idx)), (row_num, col_num))


def load_real_sparse_matrix(file_name):
    with open(file_name, "rb") as f:
        content = f.read()
        # Row and col numbers.
        row_num = np_integer(struct.unpack("=q", content[:8])[0])
        col_num = np_integer(struct.unpack("=q", content[8:16])[0])
        nonzeros_num = np_integer(struct.unpack("=q", content[16:24])[0])
        row_idx = []
        col_idx = []
        data = []
        byte_cnt = 24
        for _ in range(nonzeros_num):
            r = np_integer(struct.unpack(
                "=q", content[byte_cnt:byte_cnt + 8])[0])
            byte_cnt += 8
            c = np_integer(struct.unpack(
                "=q", content[byte_cnt:byte_cnt + 8])[0])
            byte_cnt += 8
            v = np_real(struct.unpack("=d", content[byte_cnt:byte_cnt + 8])[0])
            byte_cnt += 8
            row_idx.append(r)
            col_idx.append(c)
            data.append(v)
        row_idx = to_integer_array(row_idx)
        col_idx = to_integer_array(col_idx)
        data = to_real_array(data)
        # CSR matrix format.
        return csr_array((data, (row_idx, col_idx)), (row_num, col_num))


def save_real_sparse_matrix(file_name, mat):
    triplets = sparse_matrix_to_triplets(mat)
    with open(file_name, "wb") as f:
        # Write row and col numbers.
        row_num, col_num = mat.shape
        row_num = np_integer(row_num)
        col_num = np_integer(col_num)
        nonzeros_num = mat.nnz
        nonzeros_num = np_integer(nonzeros_num)
        f.write(struct.pack("=q", row_num))
        f.write(struct.pack("=q", col_num))
        f.write(struct.pack("=q", nonzeros_num))

        for r, c, v in triplets:
            f.write(struct.pack("=q", np_integer(r)))
            f.write(struct.pack("=q", np_integer(c)))
            f.write(struct.pack("=d", np_real(v)))


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
    Get fin control vertices, size 2N + 2
    (2N ray + penducle)
    
    :param verts: Vertex positions
    :param int: num of rays
    """
    ctrl_idx = np.empty(2 * N + 2, dtype=int)

    z_fin_ant = np.linspace(FIN_ANT_LB[1], FIN_ANT_UB[1], N)
    z_fin_post = np.linspace(FIN_POST_LB[1], FIN_POST_UB[1], N)       # fin posterior height
    for i in range(N):
        ctrl_idx[i], _ = find_closest_vertex(verts, [FIN_ANT_LB[0], 0, z_fin_ant[i]])
        ctrl_idx[i + N], _ = find_min_x_at_z(verts, z_fin_post[i])
    
    # penducle vertices
    ctrl_idx[-2], _ = find_closest_vertex(verts, [0, 0, PENDUCLE_LB])
    ctrl_idx[-1], _ = find_closest_vertex(verts, [0, 0, PENDUCLE_UB])

    return ctrl_idx


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


def expand_action_range(val, n):
    """
    Expand action range to a list
    """
    if isinstance(val, (int, float)):
        return [val] * n
    elif isinstance(val, (list, tuple)):
        if len(val) != n:
            raise ValueError("Action range incorrect size!")
        return list(val)
    else:
        raise TypeError("Incorrect type for action range, should be scalar, list or tuple.")
    