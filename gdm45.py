import json
import os
from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
from PIL import Image
import scipy


@dataclass(frozen=True)
class Ray:
    start: tuple[int, int]
    end: tuple[int, int]


def as_binary(image: np.ndarray) -> np.ndarray:
    return (np.asarray(image) > 0).astype(np.uint8)


def get_pixel(image: np.ndarray, r: int, c: int, default: int = 0) -> int:
    h, w = image.shape
    if 0 <= r < h and 0 <= c < w:
        return int(image[r, c])
    return default


def neighbor_offsets():
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_neighbors(r, c, h, w):
    for dr, dc in neighbor_offsets():
        rr, cc = (r + dr, c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            yield (rr, cc)


def canonical_cycle(cycle):
    cycle = list(cycle)
    if not cycle:
        return tuple()
    reps = []
    for seq in (cycle, cycle[::-1]):
        for k in range(len(cycle)):
            reps.append(tuple(seq[k:] + seq[:k]))
    return min(reps)


def extract_boundary_segments(binary_image: np.ndarray):
    img = as_binary(binary_image)
    h, w = img.shape
    segments = []
    if h > 1:
        rs, cs = np.nonzero((img[1:, :] == 1) & (img[:-1, :] == 0))
        segments.extend(
            [((int(r + 1), int(c)), (int(r + 1), int(c + 1))) for r, c in zip(rs, cs)]
        )
        rs, cs = np.nonzero((img[:-1, :] == 1) & (img[1:, :] == 0))
        segments.extend(
            [((int(r + 1), int(c)), (int(r + 1), int(c + 1))) for r, c in zip(rs, cs)]
        )
    if w > 1:
        rs, cs = np.nonzero((img[:, 1:] == 1) & (img[:, :-1] == 0))
        segments.extend(
            [((int(r), int(c + 1)), (int(r + 1), int(c + 1))) for r, c in zip(rs, cs)]
        )
        rs, cs = np.nonzero((img[:, :-1] == 1) & (img[:, 1:] == 0))
        segments.extend(
            [((int(r), int(c + 1)), (int(r + 1), int(c + 1))) for r, c in zip(rs, cs)]
        )
    return sorted(set(segments))


def build_boundary_graph(segments):
    G = nx.Graph()
    G.add_edges_from(segments)
    return G


def checkerboard_type_at_vertex(binary_image: np.ndarray, v: tuple[int, int]):
    r, c = v
    h, w = binary_image.shape
    if not (1 <= r < h and 1 <= c < w):
        return None
    block = np.asarray(
        [
            [binary_image[r - 1, c - 1], binary_image[r - 1, c]],
            [binary_image[r, c - 1], binary_image[r, c]],
        ],
        dtype=np.uint8,
    )
    if np.array_equal(block, np.array([[0, 1], [1, 0]], dtype=np.uint8)):
        return "01_10"
    if np.array_equal(block, np.array([[1, 0], [0, 1]], dtype=np.uint8)):
        return "10_01"
    return None


def is_corner_vertex(G: nx.Graph, v: tuple[int, int], binary_image: np.ndarray) -> bool:
    deg = G.degree(v)
    if deg == 4:
        return checkerboard_type_at_vertex(binary_image, v) is not None
    if deg != 2:
        return False
    n1, n2 = list(G.neighbors(v))
    d1 = (n1[0] - v[0], n1[1] - v[1])
    d2 = (n2[0] - v[0], n2[1] - v[1])
    if d1 == (-d2[0], -d2[1]):
        return False
    is_hv_turn = (
        d1[0] == 0
        and abs(d1[1]) == 1
        and (abs(d2[0]) == 1)
        and (d2[1] == 0)
        or (abs(d1[0]) == 1 and d1[1] == 0 and (d2[0] == 0) and (abs(d2[1]) == 1))
    )
    return is_hv_turn


def find_lattice_corners(binary_image: np.ndarray):
    segments = extract_boundary_segments(binary_image)
    boundary_graph = build_boundary_graph(segments)
    corners = []
    corners_NE = []
    corners_SW = []
    corners_triangle = []
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
        ne_ok = get_pixel(binary_image, r - 1, c) != 0
        sw_ok = get_pixel(binary_image, r, c - 1) != 0
        if ne_ok:
            corners_NE.append(v)
        if sw_ok:
            corners_SW.append(v)
        if not ne_ok and (not sw_ok):
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
    return (min(r, r + di), min(c, c + dj))


def project_ray_diagonal(image, start_corners, all_corners, di, dj):
    all_corner_set = set(all_corners)
    rays = []
    for start in start_corners:
        current = start
        moved = False
        while True:
            pr, pc = crossed_pixel_for_diagonal_step(current, di, dj)
            if get_pixel(image, pr, pc) == 0:
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
    return [
        Ray(start=ray.end, end=ray.start)
        for ray in project_ray_diagonal(image, corners_SW, all_corners, di=+1, dj=-1)
    ]


def deduplicate_rays(rays):
    seen = set()
    unique = []
    for ray in rays:
        if ray.start == ray.end:
            continue
        key = tuple(sorted((ray.start, ray.end)))
        if key not in seen:
            seen.add(key)
            unique.append(Ray(start=key[0], end=key[1]))
    return unique


def scan_connected_regions(img, value):
    h, w = img.shape
    visited = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if img[r, c] != value or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            pixels = []
            touches_border = False
            while q:
                x, y = q.popleft()
                pixels.append((x, y))
                touches_border |= x == 0 or x == h - 1 or y == 0 or (y == w - 1)
                for xx, yy in get_neighbors(x, y, h, w):
                    if img[xx, yy] == value and (not visited[xx, yy]):
                        visited[xx, yy] = True
                        q.append((xx, yy))
            regions.append({"pixels": pixels, "touches_border": touches_border})
    return regions


def find_foreground_components(binary_image):
    img = as_binary(binary_image)
    h, w = img.shape
    label_image = np.zeros((h, w), dtype=int)
    components = []
    for region in scan_connected_regions(img, value=1):
        pixels = region["pixels"]
        component_id = len(components) + 1
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        label_image[rows, cols] = component_id
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[rows, cols] = 1
        components.append(
            {
                "id": component_id,
                "mask": mask,
                "bbox": (min(rows), max(rows), min(cols), max(cols)),
                "centroid": (float(np.mean(rows)), float(np.mean(cols))),
                "area": len(pixels),
            }
        )
    return (components, label_image)


def public_component_metadata(components):
    return [
        {
            "id": comp["id"],
            "bbox": comp["bbox"],
            "centroid": comp["centroid"],
            "area": comp["area"],
        }
        for comp in components
    ]


def find_holes(binary_image):
    img = as_binary(binary_image)
    h, w = img.shape
    label_image = np.zeros((h, w), dtype=int)
    holes = []
    for region in scan_connected_regions(img, value=0):
        if region["touches_border"]:
            continue
        pixels = region["pixels"]
        hole_id = len(holes) + 1
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        label_image[rows, cols] = hole_id
        holes.append(
            {
                "id": hole_id,
                "bbox": (min(rows), max(rows), min(cols), max(cols)),
                "centroid": (float(np.mean(rows)), float(np.mean(cols))),
                "area": len(pixels),
            }
        )
    return (holes, label_image)


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
            touched.update(
                (nb for nb in boundary_graph.neighbors(x) if nb in selected_nodes)
            )
        if len(touched) == 2:
            G.add_edge(*sorted(touched), kind="boundary")
        elif len(touched) > 2:
            ambiguous_components.append((set(comp), touched))
    return (G, ambiguous_components)


def add_ray_edges(G: nx.Graph, rays):
    for ray in rays:
        if ray.start not in G or ray.end not in G or ray.start == ray.end:
            continue
        if G.has_edge(ray.start, ray.end):
            if G[ray.start][ray.end].get("kind") != "ray":
                G[ray.start][ray.end]["kind"] = "boundary+ray"
        else:
            G.add_edge(ray.start, ray.end, kind="ray")


def graph_to_adjacency_matrix(G: nx.Graph):
    nodes = sorted(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    for u, v in G.edges():
        i, j = (idx[u], idx[v])
        A[i, j] = 1
        A[j, i] = 1
    return (nodes, A)


def build_adjacency_lists(A: np.ndarray):
    return [np.flatnonzero(A[u]).tolist() for u in range(A.shape[0])]


def bfs_shortest_path(
    adj_list, source: int, target: int, forbidden=None, banned_edge=None
):
    if source == target:
        return [source]
    forbidden = set() if forbidden is None else set(forbidden)
    if source in forbidden or target in forbidden:
        return None
    banned = None if banned_edge is None else tuple(sorted(banned_edge))
    parent = [-1] * len(adj_list)
    visited = np.zeros(len(adj_list), dtype=bool)
    q = deque([source])
    visited[source] = True
    while q:
        u = q.popleft()
        for v in adj_list[u]:
            if banned is not None and tuple(sorted((u, v))) == banned:
                continue
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
                return path[::-1]
            q.append(v)
    return None


def is_valid_cycle_indices(adj_set, cycle):
    if len(cycle) < 3 or len(set(cycle)) != len(cycle):
        return False
    return all(
        (cycle[(i + 1) % len(cycle)] in adj_set[cycle[i]] for i in range(len(cycle)))
    )


def is_chordless_cycle_graph(G: nx.Graph, cycle):
    if len(cycle) < 3:
        return False
    cycle_set = set(cycle)
    prev_of = {cycle[i]: cycle[i - 1] for i in range(len(cycle))}
    next_of = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}
    for u in cycle:
        for v in G.neighbors(u):
            if v in cycle_set and v != prev_of[u] and (v != next_of[u]):
                return False
    return True


def shortest_cycles_through_start(adj_list, adj_set, start: int):
    nbrs = adj_list[start]
    if len(nbrs) < 2:
        return []
    best_len = None
    best_cycles = set()
    for i, u in enumerate(nbrs):
        for v in nbrs[i + 1 :]:
            path = bfs_shortest_path(adj_list, u, v, forbidden={start})
            if path is None:
                continue
            cycle = [start] + path
            if not is_valid_cycle_indices(adj_set, cycle):
                continue
            key = canonical_cycle(cycle)
            if best_len is None or len(key) < best_len:
                best_len = len(key)
                best_cycles = {key}
            elif len(key) == best_len:
                best_cycles.add(key)
    return [list(c) for c in sorted(best_cycles)]


def find_minimal_cycles_bfs(adj_list, adj_set, triangle_start_indices=None):
    n = len(adj_list)
    triangle_start_indices = (
        [] if triangle_start_indices is None else list(triangle_start_indices)
    )
    triangle_start_indices = sorted(
        set((idx for idx in triangle_start_indices if 0 <= idx < n))
    )
    triangle_start_set = set(triangle_start_indices)
    cycle_map = {}
    processed_starts = set()
    used_starts = []
    potential_starts = set()

    def process_start(s):
        if s in processed_starts:
            return
        processed_starts.add(s)
        cycles = shortest_cycles_through_start(adj_list, adj_set, s)
        if not cycles:
            return
        used_starts.append(s)
        for cycle in cycles:
            key = canonical_cycle(cycle)
            cycle_map[key] = list(key)
            for node in cycle:
                for nb in adj_list[node]:
                    if nb not in processed_starts and nb not in triangle_start_set:
                        potential_starts.add(nb)

    for s in triangle_start_indices:
        process_start(s)
    for s in sorted(potential_starts):
        process_start(s)
    for s in range(n):
        process_start(s)
    cycles = [cycle_map[k] for k in sorted(cycle_map.keys(), key=lambda x: (len(x), x))]
    return (cycles, used_starts, sorted(potential_starts))


def find_edge_minimal_cycles_bfs(adj_list, adj_set, edge_list):
    cycle_map = {}
    for u, v in edge_list:
        path = bfs_shortest_path(adj_list, u, v, banned_edge=(u, v))
        if path is None or not is_valid_cycle_indices(adj_set, path):
            continue
        key = canonical_cycle(path)
        cycle_map[key] = list(key)
    return [cycle_map[k] for k in sorted(cycle_map.keys(), key=lambda x: (len(x), x))]


def remove_closing_duplicate(polygon):
    polygon = [tuple(p) for p in polygon]
    if len(polygon) >= 2 and polygon[0] == polygon[-1]:
        polygon = polygon[:-1]
    return polygon


def remove_consecutive_duplicates(polygon):
    polygon = remove_closing_duplicate(polygon)
    if not polygon:
        return []
    cleaned = [polygon[0]]
    for p in polygon[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return cleaned


def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def are_collinear(a, b, c):
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    return cross(v1, v2) == 0


def simplify_polygon(polygon):
    pts = remove_consecutive_duplicates(polygon)
    if len(pts) <= 3:
        return pts
    changed = True
    while changed and len(pts) > 3:
        changed = False
        new_pts = []
        for i, p in enumerate(pts):
            if are_collinear(pts[i - 1], p, pts[(i + 1) % len(pts)]):
                changed = True
                continue
            new_pts.append(p)
        pts = new_pts
    return pts


def polygon_signed_area(polygon):
    pts = remove_closing_duplicate(polygon)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i, (r1, c1) in enumerate(pts):
        r2, c2 = pts[(i + 1) % len(pts)]
        area += c1 * r2 - c2 * r1
    return 0.5 * area


def polygon_area(cycle):
    return abs(polygon_signed_area(simplify_polygon(cycle)))


def polygon_contains_hole_pixel(hole_mask, polygon):
    polygon = simplify_polygon(polygon)
    if len(polygon) < 3 or not np.any(hole_mask):
        return False

    hole_rows, hole_cols = np.nonzero(hole_mask)
    row_centers = hole_rows + 0.5
    col_centers = hole_cols + 0.5

    rows = [row for row, _ in polygon]
    cols = [col for _, col in polygon]
    candidates = (
        (row_centers >= min(rows))
        & (row_centers <= max(rows))
        & (col_centers >= min(cols))
        & (col_centers <= max(cols))
    )
    if not np.any(candidates):
        return False

    path = Path([(col, row) for row, col in polygon], closed=True)
    centers = np.column_stack((col_centers[candidates], row_centers[candidates]))
    return bool(np.any(path.contains_points(centers)))


def is_polygon_valid(hole_mask, polygon):
    simplified = simplify_polygon(polygon)
    shape_type = classify_polygon(simplified)

    if shape_type is None:
        return False, {
            "reason": "not_triangle_parallelogram_or_trapezoid",
            "simplified_polygon": simplified,
            "shape_type": None,
        }

    if polygon_contains_hole_pixel(hole_mask, simplified):
        return False, {
            "reason": "polygon_contains_hole_pixel",
            "simplified_polygon": simplified,
            "shape_type": shape_type,
        }

    return True, {
        "reason": "valid",
        "simplified_polygon": simplified,
        "shape_type": shape_type,
    }


def filter_valid_polygons(polygons, hole_mask):
    valid_polygons = []
    valid_infos = []
    rejected_infos = []
    cache = {}

    for index, polygon in enumerate(polygons):
        simplified = simplify_polygon(polygon)
        key = canonical_cycle(simplified) if len(simplified) >= 3 else tuple(simplified)

        if key not in cache:
            cache[key] = is_polygon_valid(hole_mask, simplified)

        valid, cached_info = cache[key]
        info = {
            **cached_info,
            "original_index": index,
            "original_polygon": polygon,
        }

        if valid:
            valid_polygons.append(info["simplified_polygon"])
            valid_infos.append(info)
        else:
            rejected_infos.append(info)

    return valid_polygons, valid_infos, rejected_infos


def edge_type_and_level(p, q):
    r1, c1 = p
    r2, c2 = q
    if r1 == r2:
        return ("H", r1)
    if c1 == c2:
        return ("V", c1)
    dr = r2 - r1
    dc = c2 - c1
    if dr == -dc:
        return ("D1", r1 + c1)
    if dr == dc:
        return ("D2", r1 - c1)
    raise ValueError(f"Unsupported edge direction: {p} -> {q}")


def classify_polygon(cycle):
    pts = simplify_polygon(cycle)
    if len(pts) < 3 or polygon_signed_area(pts) == 0:
        return None
    if len(pts) == 3:
        return "triangle"
    if len(pts) != 4:
        return None
    edge_types = [edge_type_and_level(pts[i], pts[(i + 1) % 4])[0] for i in range(4)]
    pair1_parallel = edge_types[0] == edge_types[2]
    pair2_parallel = edge_types[1] == edge_types[3]
    if pair1_parallel and pair2_parallel:
        return "parallelogram"
    if pair1_parallel or pair2_parallel:
        return "trapezoid"
    return None


def is_half_pixel_triangle(cycle):
    pts = simplify_polygon(cycle)
    return len(pts) == 3 and polygon_area(pts) == 0.5


def quadrilateral_width_lattice(cycle):
    pts = simplify_polygon(cycle)
    if len(pts) != 4:
        return None
    edges = [edge_type_and_level(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    widths = [
        abs(edges[i][1] - edges[i + 2][1])
        for i in (0, 1)
        if edges[i][0] == edges[i + 2][0]
    ]
    return min(widths) if widths else None


def count_decomposition_shapes(info):
    counts = {
        "triangles": 0,
        "parallelograms": 0,
        "trapezoids": 0,
        "other": 0,
        "half_pixel_triangles": 0,
        "one_pixel_wide_parallelograms": 0,
        "one_pixel_wide_trapezoids": 0,
    }
    unique = {}
    for item in info["cycles_info"]:
        cycle = simplify_polygon(item["cycle"])
        if len(cycle) >= 3:
            unique[canonical_cycle(cycle)] = cycle
    cycles_dict = []
    for i, cycle in enumerate(unique.values()):
        cls = classify_polygon(cycle)
        cycles_dict.append({"id": i, "cycle": cycle, "classification": cls})
        if cls == "triangle":
            counts["triangles"] += 1
            counts["half_pixel_triangles"] += int(is_half_pixel_triangle(cycle))
        elif cls == "parallelogram":
            counts["parallelograms"] += 1
            counts["one_pixel_wide_parallelograms"] += int(
                quadrilateral_width_lattice(cycle) == 1
            )
        elif cls == "trapezoid":
            counts["trapezoids"] += 1
            counts["one_pixel_wide_trapezoids"] += int(
                quadrilateral_width_lattice(cycle) == 1
            )
        else:
            counts["other"] += 1
    return (cycles_dict, counts)


def attach_component_metadata(info, components, component_label_image):
    info["connected_components"] = public_component_metadata(components)
    info["component_label_image"] = component_label_image
    info["component_count"] = len(components)
    return info


def tag_and_extend(dst_list, src_list, **extra_fields):
    for item in src_list:
        dst_list.append({**item, **extra_fields})


def build_cycles_info_idx(cycles_info, node_to_idx, include_component_id=False):
    cycles_info_idx = []
    for item in cycles_info:
        cycle = item["cycle"]
        if all((node in node_to_idx for node in cycle)):
            cycle_idx = [node_to_idx[node] for node in cycle]
            entry = {
                "start": cycle_idx[0],
                "cycle": cycle_idx,
                "source": item.get("source", "bfs"),
            }
            if include_component_id:
                entry["component_id"] = item["component_id"]
            cycles_info_idx.append(entry)
    return cycles_info_idx


def build_augmented_lattice_graph_single_component(binary_image: np.ndarray):
    binary_image = as_binary(binary_image)
    (
        corners,
        corners_NE,
        corners_SW,
        corners_triangle,
        corners_checker_01_10,
        corners_checker_10_01,
        segments,
        boundary_graph,
    ) = find_lattice_corners(binary_image)
    rays = deduplicate_rays(
        project_NE_ray(binary_image, corners_NE, corners)
        + project_SW_ray(binary_image, corners_SW, corners)
    )
    selected_nodes = set(corners)
    for ray in rays:
        selected_nodes.update(
            (node for node in (ray.start, ray.end) if node in boundary_graph.nodes())
        )
    G, ambiguous_components = build_selected_boundary_graph(
        boundary_graph, selected_nodes
    )
    add_ray_edges(G, rays)
    nodes, A = graph_to_adjacency_matrix(G)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    holes, hole_label_image = find_holes(binary_image)
    triangle_start_indices = [
        node_to_idx[v] for v in corners_triangle if v in node_to_idx
    ]
    adj_list = build_adjacency_lists(A)
    adj_set = [set(nbrs) for nbrs in adj_list]
    edge_list_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    triangle_seed_cycles_idx, used_idx, potential_idx = find_minimal_cycles_bfs(
        adj_list, adj_set, triangle_start_indices=triangle_start_indices
    )
    edge_cycles_idx = find_edge_minimal_cycles_bfs(adj_list, adj_set, edge_list_idx)
    cycle_idx_map = {}
    for cycle_idx in triangle_seed_cycles_idx + edge_cycles_idx:
        if len(cycle_idx) >= 3:
            key = canonical_cycle(cycle_idx)
            cycle_idx_map[key] = list(key)
    kept_cycles = []
    for cycle_idx in sorted(cycle_idx_map.keys(), key=lambda x: (len(x), x)):
        cycle_nodes = [nodes[i] for i in cycle_idx]
        if is_chordless_cycle_graph(G, cycle_nodes):
            kept_cycles.append(cycle_nodes)
    hole_mask = hole_label_image > 0
    valid_polygons, valid_infos, rejected_infos = filter_valid_polygons(
        kept_cycles, hole_mask
    )
    cycles_info = [
        {"start": cycle[0], "cycle": cycle, "source": "bfs"} for cycle in valid_polygons
    ]
    cycles_info_idx = build_cycles_info_idx(cycles_info, node_to_idx)
    removed_invalid_faces = [
        info["simplified_polygon"]
        for info in rejected_infos
        if len(info.get("simplified_polygon", [])) >= 3
    ]
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
        "hole_mask": hole_mask,
        "valid_polygon_infos": valid_infos,
        "rejected_polygon_infos": rejected_infos,
        "removed_invalid_faces": removed_invalid_faces,
        "cycles_info": cycles_info,
        "used": [nodes[i] for i in used_idx if 0 <= i < len(nodes)],
        "potential": [nodes[i] for i in potential_idx if 0 <= i < len(nodes)],
        "node_to_idx": node_to_idx,
        "used_idx": used_idx,
        "potential_idx": potential_idx,
        "cycles_info_idx": cycles_info_idx,
    }
    return (nodes, A, G, info)


def build_augmented_lattice_graph(binary_image: np.ndarray):
    binary_image = as_binary(binary_image)
    components, component_label_image = find_foreground_components(binary_image)
    if len(components) <= 1:
        nodes, A, G, info = build_augmented_lattice_graph_single_component(binary_image)
        info = attach_component_metadata(info, components, component_label_image)
        return (nodes, A, G, info)
    combined_G = nx.Graph()
    combined_boundary_graph = nx.Graph()
    combined_hole_label_image = np.zeros(binary_image.shape, dtype=int)
    combined_hole_mask = np.zeros(binary_image.shape, dtype=bool)
    segments = set()
    corners = set()
    corners_NE = set()
    corners_SW = set()
    corners_triangle = set()
    corners_checker_01_10 = set()
    corners_checker_10_01 = set()
    rays = []
    ambiguous_components = []
    holes = []
    valid_infos = []
    rejected_infos = []
    removed_invalid_faces = []
    cycles_info = []
    used = set()
    potential = set()
    next_hole_id = 1
    for component in components:
        _, _, component_G, component_info = (
            build_augmented_lattice_graph_single_component(component["mask"])
        )
        combined_G = nx.compose(combined_G, component_G)
        combined_boundary_graph = nx.compose(
            combined_boundary_graph, component_info["boundary_graph"]
        )
        segments.update(component_info["segments"])
        corners.update(component_info["corners"])
        corners_NE.update(component_info["corners_NE"])
        corners_SW.update(component_info["corners_SW"])
        corners_triangle.update(component_info["corners_triangle"])
        corners_checker_01_10.update(component_info["corners_checker_01_10"])
        corners_checker_10_01.update(component_info["corners_checker_10_01"])
        rays.extend(component_info["rays"])
        ambiguous_components.extend(component_info["ambiguous_components"])
        removed_invalid_faces.extend(component_info["removed_invalid_faces"])
        combined_hole_mask |= component_info["hole_mask"]
        used.update(component_info["used"])
        potential.update(component_info["potential"])
        for hole in component_info["holes"]:
            hole_copy = dict(hole)
            hole_copy["id"] = next_hole_id
            holes.append(hole_copy)
            next_hole_id += 1
            r_min, r_max, c_min, c_max = hole_copy["bbox"]
            component_holes = (
                component_info["hole_label_image"][r_min : r_max + 1, c_min : c_max + 1]
                > 0
            )
            combined_hole_label_image[r_min : r_max + 1, c_min : c_max + 1][
                component_holes
            ] = hole_copy["id"]
        tag_and_extend(
            valid_infos,
            component_info["valid_polygon_infos"],
            component_id=component["id"],
        )
        tag_and_extend(
            rejected_infos,
            component_info["rejected_polygon_infos"],
            component_id=component["id"],
        )
        tag_and_extend(
            cycles_info, component_info["cycles_info"], component_id=component["id"]
        )
    nodes, A = graph_to_adjacency_matrix(combined_G)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    cycles_info_idx = build_cycles_info_idx(
        cycles_info, node_to_idx, include_component_id=True
    )
    info = {
        "segments": sorted(segments),
        "boundary_graph": combined_boundary_graph,
        "corners": sorted(corners),
        "corners_NE": sorted(corners_NE),
        "corners_SW": sorted(corners_SW),
        "corners_triangle": sorted(corners_triangle),
        "corners_checker_01_10": sorted(corners_checker_01_10),
        "corners_checker_10_01": sorted(corners_checker_10_01),
        "rays": deduplicate_rays(rays),
        "ambiguous_components": ambiguous_components,
        "holes": holes,
        "hole_label_image": combined_hole_label_image,
        "hole_mask": combined_hole_mask,
        "valid_polygon_infos": valid_infos,
        "rejected_polygon_infos": rejected_infos,
        "removed_invalid_faces": removed_invalid_faces,
        "cycles_info": cycles_info,
        "used": sorted(used),
        "potential": sorted(potential),
        "node_to_idx": node_to_idx,
        "used_idx": [node_to_idx[node] for node in sorted(used) if node in node_to_idx],
        "potential_idx": [
            node_to_idx[node] for node in sorted(potential) if node in node_to_idx
        ],
        "cycles_info_idx": cycles_info_idx,
    }
    info = attach_component_metadata(info, components, component_label_image)
    return (nodes, A, combined_G, info)


def visualize_cycles(
    file_name,
    binary_image,
    cycles_dict,
    seed=None,
    visualize=True,
    save_png=False,
    output_dir="Results45",
    dpi=600,
):
    rng = np.random.default_rng(seed)
    colors = [
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
    rng.shuffle(colors)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")
    for idx, item in enumerate(cycles_dict):
        cycle = item["cycle"]
        if len(cycle) < 3:
            continue
        pts = np.array([(p[1] - 0.5, p[0] - 0.5) for p in cycle], dtype=float)
        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor=colors[idx % len(colors)],
                edgecolor="none",
                linewidth=0,
            )
        )
    ax.set_title(f"Decomposed image to {len(cycles_dict)} polygons")
    ax.set_axis_off()
    fig.tight_layout()
    png_path = None
    if save_png:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(str(file_name)))[0].replace(
            " ", "_"
        )
        png_path = os.path.join(output_dir, f"{base_name}_gdm45.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if visualize:
        plt.show()
    else:
        plt.close(fig)
    return png_path


def save_cycles_dict_json(image_file_name, cycles_dict, counts, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(str(image_file_name)))[0].replace(
        " ", "_"
    )
    json_path = os.path.join(output_dir, f"{base_name}_45decomp.json")
    data = {
        "source_image": str(image_file_name),
        "number_of_polygons": len(cycles_dict),
        "number_by_classes": counts,
        "cycles_dict": [
            {
                "id": item["id"],
                "classification": item["classification"],
                "cycle": [list(vertex) for vertex in item["cycle"]],
            }
            for item in cycles_dict
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return json_path


def single_image_decomp(
    binary_image,
    file_name="test_image",
    save_to_json=False,
    visualize=True,
    save_png=False,
    json_output_dir="Decompositions45",
    png_output_dir="Results45",
    verbose=False,
    decomposition_angle=45
):
    nodes, A, G, info = build_augmented_lattice_graph(binary_image)
    cycles_dict, counts = count_decomposition_shapes(info)
    file_stem = os.path.splitext(os.path.basename(str(file_name)))[0]
    png_path = None
    if decomposition_angle == 45:
        pass
    elif decomposition_angle == 135:
        binary_image = rotate_image(binary_image, angle=90, interpolation=0)
    else:
        raise ValueError("decomposition_angle must be either 45 or 135 degrees.")

    if visualize or save_png:
        png_path = visualize_cycles(
            file_stem,
            binary_image,
            cycles_dict,
            seed=42,
            visualize=visualize,
            save_png=save_png,
            output_dir=png_output_dir,
        )
    json_path = None
    if save_to_json:
        json_path = save_cycles_dict_json(
            file_name, cycles_dict, counts, output_dir=json_output_dir
        )
    if verbose:
        for i, item in enumerate(cycles_dict):
            print(
                f'Polygon {i} class: {item["classification"]}, nodes: {item["cycle"]}'
            )
        print(f"{file_name}: {len(cycles_dict)} polygons")
        print(counts)
        if json_path:
            print(f"JSON: {json_path}")
        if png_path:
            print(f"PNG: {png_path}")
    return {
        "nodes": nodes,
        "A": A,
        "G": G,
        "info": info,
        "cycles_dict": cycles_dict,
        "counts": counts,
        "json_path": json_path,
        "png_path": png_path,
    }


def batch_process_images(
    directory,
    save_to_json=False,
    visualize=True,
    save_png=False,
    json_output_dir="Decompositions45",
    png_output_dir="Results45",
    verbose=False,
    decomposition_angle=45,
):
    results = []
    for file_name in sorted(os.listdir(directory)):
        if not file_name.lower().endswith((".tif", ".tiff")):
            continue
        full_path = os.path.join(directory, file_name)
        binary_image = as_binary(np.array(Image.open(full_path)))
        if decomposition_angle == 45:
            pass
        elif decomposition_angle == 135:
            binary_image = rotate_image(binary_image, angle=90, interpolation=0)
        else:
            raise ValueError("decomposition_angle must be either 45 or 135 degrees.")
        
        results.append(
            single_image_decomp(
                binary_image,
                file_name=full_path,
                save_to_json=save_to_json,
                visualize=visualize,
                save_png=save_png,
                json_output_dir=json_output_dir,
                png_output_dir=png_output_dir,
                verbose=verbose,
            )
        )
    return results


def rotate_image(image, angle, interpolation):
    return scipy.ndimage.rotate(image, angle, reshape=True, order=interpolation)


if __name__ == "__main__":
    """file_name = "TestImages/phantom_3.tif"
    image = Image.open(file_name)
    array_image = np.array(image)
    binary_image = as_binary(array_image)

    single_image_decomp(
        binary_image,
        file_name=file_name,
        save_to_json=False,
        visualize=True,
        save_png=False,
        verbose=True,
        decomposition_angle=135
    )"""

    directory = "TestImages"
    batch_process_images(
        directory,
        save_to_json=True,
        visualize=False,
        save_png=True,
        json_output_dir="Decompositions135",
        png_output_dir="Results135",
        verbose=True,
        decomposition_angle=135,
    )
