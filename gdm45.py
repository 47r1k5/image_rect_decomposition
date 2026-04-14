import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from matplotlib.patches import Polygon
from matplotlib.path import Path
from collections import deque
from PIL import Image
import os
from matplotlib.colors import hsv_to_rgb

@dataclass(frozen=True)
class Ray:
    start: tuple[int, int]
    end: tuple[int, int]

def inside_pixel(image: np.ndarray, r: int, c: int) -> bool:
    h, w = image.shape
    return 0 <= r < h and 0 <= c < w

def get_pixel(image: np.ndarray, r: int, c: int, default: int = 0) -> int:
    if inside_pixel(image, r, c):
        return int(image[r, c])
    return default

def extract_boundary_segments(binary_image: np.ndarray):
    img = (binary_image > 0).astype(np.uint8)
    h, w = img.shape
    segments = set()

    for r in range(h):
        for c in range(w):
            if img[r, c] != 1:
                continue

            if r - 1 >= 0 and img[r - 1, c] == 0:
                segments.add(((r, c), (r, c + 1)))

            if r + 1 < h and img[r + 1, c] == 0:
                segments.add(((r + 1, c), (r + 1, c + 1)))

            if c - 1 >= 0 and img[r, c - 1] == 0:
                segments.add(((r, c), (r + 1, c)))

            if c + 1 < w and img[r, c + 1] == 0:
                segments.add(((r, c + 1), (r + 1, c + 1)))

    return sorted(segments)


def build_boundary_graph(segments):
    G = nx.Graph()
    for a, b in segments:
        G.add_edge(a, b)
    return G

def checkerboard_type_at_vertex(binary_image: np.ndarray, v: tuple[int, int]):
    r, c = v
    h, w = binary_image.shape

    if not (1 <= r < h and 1 <= c < w):
        return None

    block = np.array([
        [binary_image[r - 1, c - 1], binary_image[r - 1, c]],
        [binary_image[r,     c - 1], binary_image[r,     c]],
    ], dtype=np.uint8)

    if np.array_equal(block, np.array([[0, 1], [1, 0]], dtype=np.uint8)):
        return "01_10"

    if np.array_equal(block, np.array([[1, 0], [0, 1]], dtype=np.uint8)):
        return "10_01"

    return None


def is_corner_vertex(G: nx.Graph, v: tuple[int, int], binary_image: np.ndarray) -> bool:

    deg = G.degree(v)

    if deg == 2:
        n1, n2 = list(G.neighbors(v))
        d1 = (n1[0] - v[0], n1[1] - v[1])
        d2 = (n2[0] - v[0], n2[1] - v[1])

        if d1 == (-d2[0], -d2[1]):
            return False

        horiz1 = d1[0] == 0 and abs(d1[1]) == 1
        vert1 = abs(d1[0]) == 1 and d1[1] == 0
        horiz2 = d2[0] == 0 and abs(d2[1]) == 1
        vert2 = abs(d2[0]) == 1 and d2[1] == 0

        return (horiz1 and vert2) or (vert1 and horiz2)

    if deg == 4:
        return checkerboard_type_at_vertex(binary_image, v) is not None

    return False


def find_lattice_corners(binary_image: np.ndarray):
    segments = extract_boundary_segments(binary_image)
    boundary_graph = build_boundary_graph(segments)

    corners = []
    corners_NE = []
    corners_SW = []
    corners_triangle = []

    # new special classes
    corners_checker_01_10 = []
    corners_checker_10_01 = []

    for v in sorted(boundary_graph.nodes()):
        if not is_corner_vertex(boundary_graph, v, binary_image):
            continue

        r, c = v
        checker_type = checkerboard_type_at_vertex(binary_image, v)

        corners.append(v)
        if checker_type == "01_10":
            corners_checker_01_10.append(v)
            corners_NE.append(v)
            corners_SW.append(v)
            continue

        if checker_type == "10_01":
            corners_checker_10_01.append(v)
            continue

        ne_ok = (get_pixel(binary_image, r - 1, c, 0) != 0)
        sw_ok = (get_pixel(binary_image, r, c - 1, 0) != 0)

        if ne_ok:
            corners_NE.append(v)
        if sw_ok:
            corners_SW.append(v)
        if (not ne_ok) and (not sw_ok):
            corners_triangle.append(v)

    return (
        corners,
        corners_NE,
        corners_SW,
        corners_triangle,
        corners_checker_01_10,
        corners_checker_10_01,
        segments,
        boundary_graph,
    )

def crossed_pixel_for_diagonal_step(v: tuple[int, int], di: int, dj: int):
    r, c = v
    nr, nc = r + di, c + dj
    return min(r, nr), min(c, nc)


def project_ray_diagonal(image, start_corners, all_corners, di, dj):
    all_corner_set = set(all_corners)
    rays = []

    for start in start_corners:
        current = start
        moved = False

        while True:
            pr, pc = crossed_pixel_for_diagonal_step(current, di, dj)

            if get_pixel(image, pr, pc, 0) == 0:
                if moved and current != start:
                    rays.append(Ray(start=start, end=current))
                break

            current = (current[0] + di, current[1] + dj)
            moved = True

            if current in all_corner_set and current != start:
                rays.append(Ray(start=start, end=current))
                break

    return rays


def project_NE_ray(image, corners_NE, all_corners):
    return project_ray_diagonal(image, corners_NE, all_corners, di=-1, dj=+1)


def project_SW_ray(image, corners_SW, all_corners):
    rays = project_ray_diagonal(image, corners_SW, all_corners, di=+1, dj=-1)
    return [Ray(start=r.end, end=r.start) for r in rays]


def deduplicate_rays(rays):
    seen = set()
    unique = []

    for r in rays:
        if r.start == r.end:
            continue
        key = tuple(sorted((r.start, r.end)))
        if key not in seen:
            seen.add(key)
            unique.append(Ray(start=key[0], end=key[1]))

    return unique

def build_selected_boundary_graph(boundary_graph: nx.Graph, selected_nodes):
    selected_nodes = set(selected_nodes) & set(boundary_graph.nodes())

    G = nx.Graph()
    G.add_nodes_from(selected_nodes)

    for u, v in boundary_graph.edges():
        if u in selected_nodes and v in selected_nodes:
            G.add_edge(u, v, kind="boundary")

    H = boundary_graph.copy()
    H.remove_nodes_from(selected_nodes)

    ambiguous_components = []

    for comp in nx.connected_components(H):
        touched = set()
        for x in comp:
            for nb in boundary_graph.neighbors(x):
                if nb in selected_nodes:
                    touched.add(nb)

        if len(touched) == 2:
            a, b = sorted(touched)
            G.add_edge(a, b, kind="boundary")
        elif len(touched) > 2:
            ambiguous_components.append((set(comp), touched))

    return G, ambiguous_components


def add_ray_edges(G: nx.Graph, rays):
    for ray in rays:
        if ray.start in G.nodes and ray.end in G.nodes and ray.start != ray.end:
            if G.has_edge(ray.start, ray.end):
                old_kind = G[ray.start][ray.end].get("kind", "")
                if old_kind != "ray":
                    G[ray.start][ray.end]["kind"] = "boundary+ray"
            else:
                G.add_edge(ray.start, ray.end, kind="ray")


def graph_to_adjacency_matrix(G: nx.Graph):
    nodes = sorted(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    A = np.zeros((n, n), dtype=np.uint8)

    for u, v in G.edges():
        i = idx[u]
        j = idx[v]
        A[i, j] = 1
        A[j, i] = 1

    return nodes, A

def get_neighbors(r, c, h, w, connectivity=4):
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def adjacency_neighbors(A: np.ndarray, u: int):
    return [v for v, connected in enumerate(A[u]) if connected]


def bfs_shortest_path_excluding(A: np.ndarray, source: int, target: int, forbidden=None):
    if source == target:
        return [source]

    n = A.shape[0]
    forbidden = set() if forbidden is None else set(forbidden)

    if source in forbidden or target in forbidden:
        return None

    parent = [-1] * n
    visited = np.zeros(n, dtype=bool)
    q = deque([source])
    visited[source] = True

    while q:
        u = q.popleft()

        for v in adjacency_neighbors(A, u):
            if v in forbidden or visited[v]:
                continue

            visited[v] = True
            parent[v] = u

            if v == target:
                path = [v]
                cur = v
                while parent[cur] != -1:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path

            q.append(v)

    return None


def bfs_shortest_path_without_edge(A: np.ndarray, source: int, target: int, banned_edge):
    banned_edge = tuple(sorted(banned_edge))

    n = A.shape[0]
    parent = [-1] * n
    visited = np.zeros(n, dtype=bool)
    q = deque([source])
    visited[source] = True

    while q:
        u = q.popleft()

        for v in adjacency_neighbors(A, u):
            if tuple(sorted((u, v))) == banned_edge:
                continue

            if visited[v]:
                continue

            visited[v] = True
            parent[v] = u

            if v == target:
                path = [v]
                cur = v
                while parent[cur] != -1:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path

            q.append(v)

    return None


def canonical_cycle_indices(cycle):
    cycle = list(cycle)
    n = len(cycle)

    if n == 0:
        return tuple()

    reps = []
    for seq in (cycle, cycle[::-1]):
        for k in range(n):
            reps.append(tuple(seq[k:] + seq[:k]))

    return min(reps)


def is_valid_cycle_indices(A: np.ndarray, cycle):
    if len(cycle) < 3:
        return False

    if len(set(cycle)) != len(cycle):
        return False

    m = len(cycle)
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        if A[u, v] == 0:
            return False

    return True


def is_chordless_cycle_graph(G: nx.Graph, cycle):
    m = len(cycle)
    if m < 3:
        return False

    boundary_edges = {
        tuple(sorted((cycle[i], cycle[(i + 1) % m])))
        for i in range(m)
    }

    for i in range(m):
        for j in range(i + 1, m):
            u = cycle[i]
            v = cycle[j]

            if not G.has_edge(u, v):
                continue

            edge = tuple(sorted((u, v)))
            if edge not in boundary_edges:
                return False

    return True


def shortest_cycles_through_start(A: np.ndarray, start: int):
    nbrs = adjacency_neighbors(A, start)
    if len(nbrs) < 2:
        return []

    best_len = None
    best_cycles = set()

    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            u = nbrs[i]
            v = nbrs[j]

            path = bfs_shortest_path_excluding(A, u, v, forbidden={start})
            if path is None:
                continue

            cycle = [start] + path

            if not is_valid_cycle_indices(A, cycle):
                continue

            key = canonical_cycle_indices(cycle)
            L = len(key)

            if best_len is None or L < best_len:
                best_len = L
                best_cycles = {key}
            elif L == best_len:
                best_cycles.add(key)

    return [list(c) for c in sorted(best_cycles)]


def find_minimal_cycles_bfs(A: np.ndarray, triangle_start_indices=None):
    n = A.shape[0]
    triangle_start_indices = [] if triangle_start_indices is None else list(triangle_start_indices)

    triangle_start_indices = sorted(set(
        idx for idx in triangle_start_indices
        if 0 <= idx < n
    ))

    cycle_map = {}
    processed_starts = set()
    used_starts = []
    potential_starts = set()

    def process_start(s):
        if s in processed_starts:
            return

        processed_starts.add(s)
        cycles = shortest_cycles_through_start(A, s)

        if not cycles:
            return

        used_starts.append(s)

        for cycle in cycles:
            key = canonical_cycle_indices(cycle)
            cycle_map[key] = list(key)

            for node in cycle:
                for nb in adjacency_neighbors(A, node):
                    if nb not in processed_starts and nb not in triangle_start_indices:
                        potential_starts.add(nb)

    for s in triangle_start_indices:
        process_start(s)

    for s in sorted(potential_starts):
        process_start(s)

    for s in range(n):
        process_start(s)

    cycles = [cycle_map[k] for k in sorted(cycle_map.keys(), key=lambda x: (len(x), x))]
    return cycles, used_starts, sorted(potential_starts)


def find_edge_minimal_cycles_bfs(A: np.ndarray):
    n = A.shape[0]
    cycle_map = {}

    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v] == 0:
                continue

            path = bfs_shortest_path_without_edge(A, u, v, banned_edge=(u, v))
            if path is None:
                continue

            if not is_valid_cycle_indices(A, path):
                continue

            key = canonical_cycle_indices(path)
            cycle_map[key] = list(key)

    return [cycle_map[k] for k in sorted(cycle_map.keys(), key=lambda x: (len(x), x))]


def find_holes(binary_image, connectivity=4):
    img = (binary_image > 0).astype(np.uint8)
    h, w = img.shape

    visited = np.zeros((h, w), dtype=bool)
    hole_label_image = np.zeros((h, w), dtype=int)
    holes = []
    hole_id = 0

    for r in range(h):
        for c in range(w):
            if img[r, c] != 0 or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            component = []
            touches_border = False

            while q:
                x, y = q.popleft()
                component.append((x, y))

                if x == 0 or x == h - 1 or y == 0 or y == w - 1:
                    touches_border = True

                for xx, yy in get_neighbors(x, y, h, w, connectivity):
                    if img[xx, yy] == 0 and not visited[xx, yy]:
                        visited[xx, yy] = True
                        q.append((xx, yy))

            if not touches_border:
                hole_id += 1

                rows = [p[0] for p in component]
                cols = [p[1] for p in component]

                mask = np.zeros((h, w), dtype=bool)
                for x, y in component:
                    mask[x, y] = True
                    hole_label_image[x, y] = hole_id

                holes.append({
                    "id": hole_id,
                    "pixels": component,
                    "mask": mask,
                    "bbox": (min(rows), max(rows), min(cols), max(cols)),
                    "centroid": (float(np.mean(rows)), float(np.mean(cols))),
                    "area": len(component),
                })

    return holes, hole_label_image

def canonical_cycle_nodes(cycle):
    cycle = list(cycle)
    n = len(cycle)

    if n == 0:
        return tuple()

    reps = []
    for seq in (cycle, cycle[::-1]):
        for k in range(n):
            reps.append(tuple(seq[k:] + seq[:k]))

    return min(reps)


def polygon_signed_area(face):
    pts = [(c, r) for (r, c) in face]
    area = 0.0
    n = len(pts)

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return 0.5 * area


def extract_bounded_faces(G: nx.Graph):
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("The augmented graph is not planar.")

    visited_half_edges = set()
    all_faces = []

    for u in embedding.nodes():
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in visited_half_edges:
                continue

            face = embedding.traverse_face(u, v, mark_half_edges=visited_half_edges)

            if len(face) >= 3:
                all_faces.append(face)

    unique_faces = {}
    for face in all_faces:
        key = canonical_cycle_nodes(face)
        unique_faces[key] = list(key)

    faces = list(unique_faces.values())

    if not faces:
        return [], []

    outer_face = max(faces, key=lambda f: abs(polygon_signed_area(f)))
    outer_key = canonical_cycle_nodes(outer_face)

    bounded_faces = [
        face for face in faces
        if canonical_cycle_nodes(face) != outer_key
    ]

    return bounded_faces, outer_face

def face_contains_hole(face, holes):
    verts = np.array([(c, r) for (r, c) in face], dtype=float)
    path = Path(verts, closed=True)

    for hole in holes:
        cr, cc = hole["centroid"]
        centroid_pt = (cc + 0.5, cr + 0.5)

        if path.contains_point(centroid_pt, radius=1e-9):
            return True

        for r, c in hole["pixels"]:
            pt = (c + 0.5, r + 0.5)
            if path.contains_point(pt, radius=1e-9):
                return True

    return False

def remove_hole_faces(faces, holes):
    kept_faces = []
    removed_faces = []

    for face in faces:
        if face_contains_hole(face, holes):
            removed_faces.append(face)
        else:
            kept_faces.append(face)

    return kept_faces, removed_faces


def remove_hole_cycles(cycles, holes):
    kept_cycles = []
    removed_cycles = []

    for cycle in cycles:
        if face_contains_hole(cycle, holes):
            removed_cycles.append(cycle)
        else:
            kept_cycles.append(cycle)

    return kept_cycles, removed_cycles


def extract_hole_boundary_segments(binary_image: np.ndarray, hole):
    img = (binary_image > 0).astype(np.uint8)
    h, w = img.shape
    segments = set()

    for r, c in hole["pixels"]:
        if r - 1 >= 0 and img[r - 1, c] == 1:
            segments.add(((r, c), (r, c + 1)))

        if r + 1 < h and img[r + 1, c] == 1:
            segments.add(((r + 1, c), (r + 1, c + 1)))

        if c - 1 >= 0 and img[r, c - 1] == 1:
            segments.add(((r, c), (r + 1, c)))

        if c + 1 < w and img[r, c + 1] == 1:
            segments.add(((r, c + 1), (r + 1, c + 1)))

    return sorted(segments)


def find_hole_boundary_cycles(binary_image: np.ndarray, holes):
    cycles = []

    for hole in holes:
        segments = extract_hole_boundary_segments(binary_image, hole)
        if not segments:
            cycles.append([])
            continue

        H = nx.Graph()
        for a, b in segments:
            H.add_edge(a, b)

        hole_cycles = []
        seen = set()
        for cycle in nx.cycle_basis(H):
            if len(cycle) < 3:
                continue

            key = canonical_cycle_nodes(cycle)
            if key not in seen:
                seen.add(key)
                hole_cycles.append(list(key))

        cycles.append(hole_cycles)

    return cycles


def build_augmented_lattice_graph(binary_image: np.ndarray):
    binary_image = (binary_image > 0).astype(np.uint8)

    corners, corners_NE, corners_SW, corners_triangle, corners_checker_01_10, corners_checker_10_01, segments, boundary_graph = (
        find_lattice_corners(binary_image)
    )

    rays_NE = project_NE_ray(binary_image, corners_NE, corners)
    rays_SW = project_SW_ray(binary_image, corners_SW, corners)
    rays = deduplicate_rays(rays_NE + rays_SW)

    selected_nodes = set(corners)
    for ray in rays:
        if ray.start in boundary_graph.nodes():
            selected_nodes.add(ray.start)
        if ray.end in boundary_graph.nodes():
            selected_nodes.add(ray.end)

    G, ambiguous_components = build_selected_boundary_graph(
        boundary_graph, selected_nodes
    )
    add_ray_edges(G, rays)

    nodes, A = graph_to_adjacency_matrix(G)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    holes, hole_label_image = find_holes(binary_image, connectivity=4)

    triangle_start_indices = [
        node_to_idx[v]
        for v in corners_triangle
        if v in node_to_idx
    ]

    triangle_seed_cycles_idx, used_idx, potential_idx = find_minimal_cycles_bfs(
        A,
        triangle_start_indices=triangle_start_indices,
    )
    edge_cycles_idx = find_edge_minimal_cycles_bfs(A)

    cycle_idx_map = {}
    for cycle_idx in triangle_seed_cycles_idx + edge_cycles_idx:
        if len(cycle_idx) < 3:
            continue
        key = canonical_cycle_indices(cycle_idx)
        cycle_idx_map[key] = list(key)

    kept_cycles = []
    for cycle_idx in sorted(cycle_idx_map.keys(), key=lambda x: (len(x), x)):
        cycle_nodes = [nodes[i] for i in cycle_idx]

        if not is_chordless_cycle_graph(G, cycle_nodes):
            continue

        kept_cycles.append(cycle_nodes)

    kept_cycles, removed_hole_faces = remove_hole_cycles(kept_cycles, holes)

    hole_boundary_cycles_per_hole = find_hole_boundary_cycles(binary_image, holes)
    existing_removed = {canonical_cycle_nodes(cycle) for cycle in removed_hole_faces}

    for hole, hole_cycles in zip(holes, hole_boundary_cycles_per_hole):
        already_removed = any(face_contains_hole(cycle, [hole]) for cycle in removed_hole_faces)
        if already_removed:
            continue

        for cycle in hole_cycles:
            key = canonical_cycle_nodes(cycle)
            if key not in existing_removed:
                removed_hole_faces.append(cycle)
                existing_removed.add(key)

    cycles_info = [
        {
            "start": cycle[0],
            "cycle": cycle,
            "source": "bfs",
        }
        for cycle in kept_cycles
    ]

    cycles_info_idx = []
    for cycle in kept_cycles:
        if all(node in node_to_idx for node in cycle):
            cycle_idx = [node_to_idx[node] for node in cycle]
            cycles_info_idx.append({
                "start": cycle_idx[0],
                "cycle": cycle_idx,
                "source": "bfs",
            })

    info = {
        "segments": segments,
        "boundary_graph": boundary_graph,
        "corners": corners,
        "corners_NE": corners_NE,
        "corners_SW": corners_SW,
        "corners_triangle": corners_triangle,
        "corners_checker_01_10": corners_checker_01_10,
        "corners_checker_10_01": corners_checker_10_01,
        "rays": rays,
        "ambiguous_components": ambiguous_components,
        "holes": holes,
        "hole_label_image": hole_label_image,
        "removed_hole_faces": removed_hole_faces,
        "cycles_info": cycles_info,
        "outer_face": [],
        "used": [nodes[i] for i in used_idx if 0 <= i < len(nodes)],
        "potential": [nodes[i] for i in potential_idx if 0 <= i < len(nodes)],
        "node_to_idx": node_to_idx,
        "used_idx": used_idx,
        "potential_idx": potential_idx,
        "cycles_info_idx": cycles_info_idx,
    }

    return nodes, A, G, info

def draw_graph_edges(ax, G, alpha=1.0):
    for u, v, data in G.edges(data=True):
        kind = data.get("kind", "")
        x = [u[1] - 0.5, v[1] - 0.5]
        y = [u[0] - 0.5, v[0] - 0.5]

        if kind == "ray":
            ax.plot(x, y, linestyle="--", linewidth=2, alpha=alpha)
        elif kind == "boundary+ray":
            ax.plot(x, y, linewidth=3, alpha=alpha)
        else:
            ax.plot(x, y, linewidth=2, alpha=alpha)


def visualize_corner_classes(binary_image, info):
    fig = plt.figure("Corner classes", figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")

    if info["corners"]:
        ys = [p[0] - 0.5 for p in info["corners"]]
        xs = [p[1] - 0.5 for p in info["corners"]]
        ax.scatter(xs, ys, s=35, c="red", label="corners")

    if info["corners_NE"]:
        ys = [p[0] - 0.5 for p in info["corners_NE"]]
        xs = [p[1] - 0.5 for p in info["corners_NE"]]
        ax.scatter(xs, ys, s=55, c="blue", label="NE")

    if info["corners_SW"]:
        ys = [p[0] - 0.5 for p in info["corners_SW"]]
        xs = [p[1] - 0.5 for p in info["corners_SW"]]
        ax.scatter(xs, ys, s=55, c="green", label="SW")

    if info["corners_triangle"]:
        ys = [p[0] - 0.5 for p in info["corners_triangle"]]
        xs = [p[1] - 0.5 for p in info["corners_triangle"]]
        ax.scatter(xs, ys, s=70, c="orange", marker="s", label="triangle corners")

    ax.set_title("Lattice corners and classes")
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()


def visualize_augmented_graph(binary_image, nodes, G, info, show_labels=True):
    fig = plt.figure("Augmented graph", figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")
    draw_graph_edges(ax, G)

    if nodes:
        ys = [p[0] - 0.5 for p in nodes]
        xs = [p[1] - 0.5 for p in nodes]
        ax.scatter(xs, ys, s=40, c="red", label="graph nodes")

    if show_labels:
        for i, (r, c) in enumerate(nodes):
            ax.text(c - 0.35, r - 0.35, str(i), fontsize=8)

    ax.set_title("Augmented graph projected onto object")
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()


def visualize_holes(binary_image, info):
    fig = plt.figure("Holes", figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")

    for hole in info["holes"]:
        ys = [p[0] for p in hole["pixels"]]
        xs = [p[1] for p in hole["pixels"]]
        ax.scatter(xs, ys, s=8, label=f'hole {hole["id"]}')

    ax.set_title("Detected holes")
    ax.set_axis_off()
    if info["holes"]:
        ax.legend()
    plt.tight_layout()


def visualize_cycles(file_name, binary_image, info, seed=None):
    rng = np.random.default_rng(seed)

    color_pool = [
        "#ff1744", 
        "#f50057", 
        "#d500f9", 
        "#651fff", 
        "#2979ff",
        "#00b0ff",
        "#00e5ff", 
        "#1de9b6",  
        "#00e676",  
        "#76ff03", 
        "#c6ff00",  
        "#ffea00",  
        "#ffc400",
        "#ff9100", 
        "#ff3d00", 
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")

    if info["cycles_info"]:
        shuffled_pool = color_pool.copy()
        rng.shuffle(shuffled_pool)

        for idx, item in enumerate(info["cycles_info"]):
            cycle = item["cycle"]

            if len(cycle) >= 3:
                pts = np.array([(p[1] - 0.5, p[0] - 0.5) for p in cycle], dtype=float)

                color = shuffled_pool[idx % len(shuffled_pool)]

                poly = Polygon(
                    pts,
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0
                )
                ax.add_patch(poly)

    ax.set_title("Decomposed image to {} polygons".format(len(info["cycles_info"])))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("Results45/" + file_name + "_gdm45.png", dpi=600, bbox_inches="tight")


def visualize_removed_hole_faces(binary_image, info):
    fig = plt.figure("Removed hole-polygons", figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")

    for face in info["removed_hole_faces"]:
        pts = np.array([(p[1] - 0.5, p[0] - 0.5) for p in face], dtype=float)
        poly = Polygon(
            pts,
            closed=True,
            facecolor="red",
            edgecolor="none",
            linewidth=0,
            alpha=0.5,
        )
        ax.add_patch(poly)

    ax.set_title("Faces removed because they are holes")
    ax.set_axis_off()
    plt.tight_layout()


def visualize_everything_separate(file_name,
    binary_image, nodes, A, G, info, show_labels=False, seed=None
):
    visualize_corner_classes(binary_image, info)
    visualize_augmented_graph(binary_image, nodes, G, info, show_labels=show_labels)
    visualize_holes(binary_image, info)
    visualize_cycles(file_name, binary_image, info, seed=seed)
    plt.show()

def single_image_decomp(file_name, binary_image):
    try:
        print(f"Processing {file_name}...")
        nodes, A, G, info = build_augmented_lattice_graph(binary_image)
        
        print("Number of polygons:", len(info["cycles_info"]))
        for i, item in enumerate(info["cycles_info"]):
            print(f'Polygon {i} nodes: {item["cycle"]}')
    except Exception as e:
        print(f"Error processing image: {e}")

    #visualize_everything_separate(file_name, binary_image, nodes, A, G, info, seed=42)
    visualize_cycles(file_name, binary_image, info, seed=42)

def batch_process_images(directory):
    try:
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if file_name.endswith(".tif"):
                file_name_no_ext = os.path.splitext(file_name)[0]
                image = Image.open(full_path)
                array_image = np.array(image)
                binary_image = np.array(array_image).astype(bool).astype(int)
                single_image_decomp(file_name_no_ext, binary_image)
    except Exception as e:
        print(f"Error processing images in directory: {e}")

def triangle_matrix(M,h=64, w=64, apex_row=8, apex_col=None, height=40, base_width=40):
    if apex_col is None:
        apex_col = w // 2

    for r in range(height):
        row = apex_row + r
        if row < 0 or row >= h:
            continue

        half = (base_width / 2) * (r / (height - 1)) if height > 1 else 0
        left = int(np.ceil(apex_col - half))
        right = int(np.floor(apex_col + half))

        left = max(left, 0)
        right = min(right, w - 1)
        if left <= right:
            M[row, left:right+1] = 1

    return M

if __name__ == "__main__":
    """
    #Test 1: 
    binary_image = np.zeros((300, 300), dtype=np.uint8)
    binary_image[53:179, 28:229] = 1
    binary_image[3:54, 78:129] = 1
    binary_image[3:54, 178:229] = 1
    binary_image[178:279, 78:129] = 1
    binary_image[228:279, 28:179] = 1
    binary_image[103:154, 228:254] = 1
    binary_image[103:129, 253:279] = 1
    binary_image[64:128, 64:128] = 1
    """
    """
    #Test 2:
    binary_image = np.zeros((300, 300), dtype=np.uint8)
    binary_image[50:250, 50:250] = 1
    binary_image[100:200, 100:200] = 0
    binary_image[100:130, 100:130] = 1
    binary_image[170:200, 100:130] = 1
    binary_image[170:200, 170:200] = 1
    binary_image[100:130, 170:200] = 1
    """
    
    """
    #Triangle test:
    binary_image = np.zeros((100, 100), dtype=np.uint8)
    binary_image = triangle_matrix(binary_image, h=100,w=100, apex_row=30, height=64, base_width=64)
    binary_image[70:80, 45:55] = 0
    """
    
    
    #Running on actual image:
    file_name = "TestImages/phantom_6.tif"
    image = Image.open(file_name)
    array_image = np.array(image)
    binary_image = np.array(array_image).astype(bool).astype(int)
    
    """
    #joint_K2
    X,Y = np.indices(binary_image.shape)
    x1, y1 = 15, 256
    x2, y2 = 256, 15
    x3, y3 = 256, 0
    x4, y4 = 0, 256
    mask1 = (Y-y1)*(x2-x1)>(X-x1)*(y2-y1)
    mask2 = (Y-y3)*(x4-x3)>(X-x3)*(y4-y3)
    binary_image[mask1]=0
    binary_image[mask2]=0"""
    
    """
    X,Y = np.indices(binary_image.shape)
    x1, y1 = 15, 256
    x2, y2 = 256, 15
    x3, y3 = 256, 70
    x4, y4 = 70, 256
    mask1 = (Y-y1)*(x2-x1)>(X-x1)*(y2-y1)
    mask2 = (Y-y3)*(x4-x3)>(X-x3)*(y4-y3)
    #binary_image[mask1]=0
    binary_image[mask2]=0
    """
    """
    binary_image[0:35, 0:256] = 0
    binary_image[80:256, 0:256] = 0
    binary_image[0:256, 0:170] = 0
    binary_image[0:256, 222:256] = 0
    """
    #single_image_decomp(file_name, binary_image)
    
    directory = "TestImages"
    batch_process_images(directory)
    