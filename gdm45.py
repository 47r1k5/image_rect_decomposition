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
    """
    A face is treated as a hole-face if it contains the center of
    at least one hole pixel component.
    """
    verts = np.array([(c, r) for (r, c) in face], dtype=float)
    path = Path(verts, closed=True)

    for hole in holes:
        r, c = hole["pixels"][0]
        pt = (c + 0.5, r + 0.5)

        if path.contains_point(pt):
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

    faces, outer_face = extract_bounded_faces(G)
    faces, removed_hole_faces = remove_hole_faces(faces, holes)

    cycles_info = [
        {
            "start": face[0],
            "cycle": face,
            "source": "face",
        }
        for face in faces
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
        "removed_hole_faces": removed_hole_faces,
        "cycles_info": cycles_info,
        "outer_face": outer_face,
        "used": [],
        "potential": [],
        "node_to_idx": node_to_idx,
        "used_idx": [],
        "potential_idx": [],
        "cycles_info_idx": [
            {
                "start": node_to_idx[item["start"]],
                "cycle": [node_to_idx[x] for x in item["cycle"] if x in node_to_idx],
                "source": item["source"],
            }
            for item in cycles_info
            if item["start"] in node_to_idx
        ],
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


def visualize_cycles(binary_image, info, seed=None):
    rng = np.random.default_rng(seed)

    fig = plt.figure("Polygons after removing holes", figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.imshow(binary_image, cmap="gray", origin="upper", interpolation="nearest")

    if info["cycles_info"]:
        for item in info["cycles_info"]:
            cycle = item["cycle"]

            if len(cycle) >= 3:
                pts = np.array([(p[1] - 0.5, p[0] - 0.5) for p in cycle], dtype=float)
                color = rng.random(3)
                h = rng.random()
                s = rng.uniform(0.75, 1.0)   # high saturation -> not pale
                v = rng.uniform(0.35, 0.85)  # avoid near-white
                color = hsv_to_rgb([h, s, v])
                    
                poly = Polygon(
                    pts,
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0                
                    )
                ax.add_patch(poly)

    ax.set_title("Kept polygons")
    ax.set_axis_off()
    plt.tight_layout()


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


def visualize_everything_separate(
    binary_image, nodes, A, G, info, show_labels=False, seed=None
):
    visualize_corner_classes(binary_image, info)
    visualize_augmented_graph(binary_image, nodes, G, info, show_labels=show_labels)
    visualize_holes(binary_image, info)
    visualize_cycles(binary_image, info, seed=seed)
    plt.show()

def single_image_decomp(binary_image):
    try:
        nodes, A, G, info = build_augmented_lattice_graph(binary_image)
        
        print("Number of polygons:", len(info["cycles_info"]))
        for i, item in enumerate(info["cycles_info"]):
            print(f'Polygon {i} nodes: {item["cycle"]}')
    except Exception as e:
        print(f"Error processing image: {e}")

    visualize_everything_separate(binary_image, nodes, A, G, info, seed=42)

def batch_process_images(directory):
    try:
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if file_name.endswith(".tif"):
                print(f"Processing {file_name}...")
                
                image = Image.open(full_path)
                array_image = np.array(image)
                binary_image = np.array(array_image).astype(bool).astype(int)
                single_image_decomp(binary_image)
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
    
    """
    #Running on actual image:
    file_name = "TestImages/joint_K7.tif"
    image = Image.open(file_name)
    array_image = np.array(image)
    binary_image = np.array(array_image).astype(bool).astype(int)
    """
    
    #single_image_decomp(binary_image)
    
    """
    directory = "TestImages"
    batch_process_images(directory)
    """