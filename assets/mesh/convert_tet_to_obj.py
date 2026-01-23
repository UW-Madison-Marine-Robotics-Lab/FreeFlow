import numpy as np
from collections import defaultdict, Counter, deque
import meshio

INFILE = "clownfish.mesh"
OUT_OBJ = "clownfish.obj"
OUT_STL = "clownfish.stl"


def read_medit_tet_no_refs(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    def find_idx(token):
        for i, ln in enumerate(lines):
            if ln.split()[0] == token:
                return i
        return -1

    iv = find_idx("Vertices")
    it = find_idx("Tetrahedra")
    if iv < 0 or it < 0:
        raise ValueError("Expected 'Vertices' and 'Tetrahedra' sections.")

    nV = int(lines[iv + 1])
    pts = []
    for k in range(nV):
        parts = lines[iv + 2 + k].split()
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    pts = np.asarray(pts, dtype=float)

    nT = int(lines[it + 1])
    tets = []
    for k in range(nT):
        parts = lines[it + 2 + k].split()
        tets.append([int(parts[0])-1, int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
    tets = np.asarray(tets, dtype=int)
    return pts, tets


def extract_boundary_faces_oriented(pts, tets):
    """
    Extract boundary triangles with a *known* orientation from tets.
    Key idea: For each tet face, choose an orientation that points outward
    relative to the tet's opposite vertex. Then cancel internal faces.
    """
    # faces with opposite vertex index for outward test
    # For a tet (a,b,c,d), the face opposite d is (a,b,c), etc.
    face_specs = [
        (np.array([0, 1, 2]), 3),
        (np.array([0, 1, 3]), 2),
        (np.array([0, 2, 3]), 1),
        (np.array([1, 2, 3]), 0),
    ]

    # map: sorted face tuple -> list of oriented faces (should be 1 for boundary, 2 for internal)
    face_map = defaultdict(list)

    for tet in tets:
        tet_pts = pts[tet]
        for f_idx, opp in face_specs:
            tri = tet[f_idx]          # 3 vertex indices
            opp_v = tet[opp]          # opposite vertex index

            a, b, c = tri
            pa, pb, pc = pts[a], pts[b], pts[c]
            popp = pts[opp_v]

            # normal based on (a,b,c)
            n = np.cross(pb - pa, pc - pa)
            # If normal points towards opposite vertex, flip to make it point outward from tet
            # We want n · (popp - pa) < 0 (opposite vertex lies "behind" the face normal)
            if np.dot(n, popp - pa) > 0:
                tri = np.array([a, c, b], dtype=int)  # flip winding

            key = tuple(sorted(tri.tolist()))
            face_map[key].append(tri)

    # boundary faces appear exactly once
    boundary_tris = []
    for key, tris in face_map.items():
        if len(tris) == 1:
            boundary_tris.append(tris[0])

    return np.asarray(boundary_tris, dtype=int)


def orient_surface_consistently(tris):
    """
    Ensure all triangles are consistently oriented relative to neighbors.
    BFS over triangle adjacency defined by shared edges.
    """
    # Build edge -> (tri_index, directed_edge) list
    edge_to_tris = defaultdict(list)
    for ti, (a, b, c) in enumerate(tris):
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            key = (min(u, v), max(u, v))
            edge_to_tris[key].append((ti, (u, v)))

    # Build adjacency: triangles sharing an edge
    adj = defaultdict(list)
    for key, items in edge_to_tris.items():
        if len(items) == 2:
            (t0, e0), (t1, e1) = items
            adj[t0].append((t1, key))
            adj[t1].append((t0, key))

    oriented = tris.copy()
    visited = np.zeros(len(tris), dtype=bool)

    for start in range(len(tris)):
        if visited[start]:
            continue
        # BFS component
        q = deque([start])
        visited[start] = True

        while q:
            t = q.popleft()
            a, b, c = oriented[t]
            # current directed edges set for quick check
            cur_edges = {(a, b), (b, c), (c, a)}

            for nb, edge_key in adj.get(t, []):
                if visited[nb]:
                    continue

                x, y, z = oriented[nb]
                nb_edges = {(x, y), (y, z), (z, x)}

                # For a shared edge (u,v), proper consistent orientation means:
                # if current has (u,v), neighbor should have (v,u).
                u, v = edge_key
                if (u, v) in cur_edges and (u, v) in nb_edges:
                    # neighbor matches direction -> flip neighbor
                    oriented[nb] = np.array([x, z, y], dtype=int)
                elif (v, u) in cur_edges and (v, u) in nb_edges:
                    oriented[nb] = np.array([x, z, y], dtype=int)

                visited[nb] = True
                q.append(nb)

    return oriented


def signed_volume(pts, tris):
    # Volume of closed oriented triangle mesh:
    # V = (1/6) * sum over tris of dot(p0, cross(p1, p2))
    p0 = pts[tris[:, 0]]
    p1 = pts[tris[:, 1]]
    p2 = pts[tris[:, 2]]
    return np.sum(np.einsum("ij,ij->i", p0, np.cross(p1, p2))) / 6.0


def main():
    pts, tets = read_medit_tet_no_refs(INFILE)

    # 1) boundary extraction with outward orientation per tet
    tris = extract_boundary_faces_oriented(pts, tets)
    if len(tris) == 0:
        raise RuntimeError("No boundary triangles found.")

    # 2) make orientation consistent across the surface
    tris = orient_surface_consistently(tris)

    # 3) ensure positive volume; if negative, flip all
    vol = signed_volume(pts, tris)
    if vol < 0:
        tris = tris[:, [0, 2, 1]]
        vol = -vol

    print(f"Boundary triangles: {len(tris)}")
    print(f"Signed volume (should be +): {vol}")

    surf = meshio.Mesh(points=pts, cells=[("triangle", tris)])
    meshio.write(OUT_OBJ, surf)
    meshio.write(OUT_STL, surf)
    print("Wrote:", OUT_OBJ, OUT_STL)


if __name__ == "__main__":
    main()