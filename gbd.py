#!/usr/bin/env python3
"""
Graph-Based Decomposition (GBD) into rectangles for *binary* images (TIFF supported),
with visualization.

Key fixes vs. the earlier version (addresses seed-dependent rectangle count):
  1) Level-1 chord candidates are restricted to *adjacent* cogrid concave vertices
     on each scanline/column. This guarantees horizontal chords don't overlap each other
     (same for vertical), so the conflict graph is truly bipartite as assumed by GBD.
  2) Level-2 processes concave vertices in a deterministic order (lexicographic).
     Randomness affects only the *direction choice* at a concave vertex (as in the paper),
     not which vertex is processed next. This is crucial for invariance of rectangle count.

Input:
  - binary image file (0/255 or 0/1) .tif/.tiff/.png/.bmp/.jpg ...
  - strict mode: requires <=2 unique grayscale values (1 is allowed for all-black/all-white)

Output:
  - rectangles: (row0, col0, height, width)
  - visualization: left = black/white mask, right = colored rectangle tiling

Usage:
  Single file:
    python gbd_binary_gbd.py --input path/to/mask.tif --save out.png

  Directory batch:
    python gbd_binary_gbd.py --test-dir ./tests --save-dir ./out --dump-summary summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image  # Pillow supports TIFF


# ----------------------------- Data structures ----------------------------- #

@dataclass(frozen=True)
class Chord:
    orient: str          # 'H' or 'V'
    fixed: int           # y for H, x for V
    a: int               # start coord along varying axis
    b: int               # end coord along varying axis


# ----------------------------- Binary image I/O ----------------------------- #

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}


def load_binary_mask(path: str, *, strict: bool = True) -> np.ndarray:
    """
    Load a binary image and return a 2D uint8 mask with values in {0,1}.
    strict=True enforces binarity after grayscale conversion.

    Foreground is interpreted as the *higher* of the two values.
    """
    img = Image.open(path)
    try:
        img.seek(0)  # first frame if multipage
    except Exception:
        pass

    img = img.convert("L")
    arr = np.array(img)

    uniq = np.unique(arr)
    if strict and uniq.size not in (1, 2):
        raise ValueError(
            f"Input is not binary: found {uniq.size} unique grayscale values. "
            f"First 10: {uniq[:10].tolist()}. Provide a true binary image (e.g., 0/255)."
        )

    if uniq.size == 1:
        return (arr > 0).astype(np.uint8)

    lo, hi = int(uniq[0]), int(uniq[-1])
    return (arr == hi).astype(np.uint8)


# ----------------------------- Concave vertices (reflex vertices) ----------------------------- #

def concave_vertices(mask01: np.ndarray) -> List[Tuple[int, int]]:
    """
    Concave grid vertices of a *rectilinear* digital polygon induced by mask.
    Grid vertex (x,y) is concave iff among its 4 incident pixels exactly 3 are inside.

    Returns list of (x,y) in vertex coordinates (x = column index, y = row index).
    """
    h, w = mask01.shape
    p = np.pad(mask01.astype(np.uint8), ((1, 1), (1, 1)), mode="constant", constant_values=0)
    NW = p[0:h+1, 0:w+1]
    NE = p[0:h+1, 1:w+2]
    SW = p[1:h+2, 0:w+1]
    SE = p[1:h+2, 1:w+2]
    s = NW + NE + SW + SE
    ys, xs = np.where(s == 3)
    return list(zip(xs.tolist(), ys.tolist()))


# ----------------------------- Interior edge tests (for chord validity) ----------------------------- #

def interior_edges(mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a mask, compute *interior* unit edges (between two inside pixels).
      interior_h[y,x] : edge (x,y)->(x+1,y) is interior iff mask[y-1,x]=1 and mask[y,x]=1
      interior_v[y,x] : edge (x,y)->(x,y+1) is interior iff mask[y,x-1]=1 and mask[y,x]=1
    """
    H, W = mask01.shape
    mh = mask01.astype(bool)

    interior_h = np.zeros((H + 1, max(W - 1, 0)), dtype=bool)
    if H >= 2 and W >= 2:
        interior_h[1:H, :] = mh[0:H-1, 0:W-1] & mh[1:H, 0:W-1]

    interior_v = np.zeros((max(H - 1, 0), W + 1), dtype=bool)
    if H >= 2 and W >= 2:
        interior_v[:, 1:W] = mh[0:H-1, 0:W-1] & mh[0:H-1, 1:W]

    return interior_h, interior_v


def prefix_sums_edges(interior_h: np.ndarray, interior_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ph = np.cumsum(interior_h.astype(np.int32), axis=1) if interior_h.size else interior_h.astype(np.int32)
    pv = np.cumsum(interior_v.astype(np.int32), axis=0) if interior_v.size else interior_v.astype(np.int32)
    return ph, pv


def all_true_h(ph: np.ndarray, y: int, x1: int, x2: int) -> bool:
    """Check interior_h[y, x1:x2] is all True."""
    if x2 <= x1:
        return False
    row = ph[y]
    s = row[x2 - 1] - (row[x1 - 1] if x1 > 0 else 0)
    return s == (x2 - x1)


def all_true_v(pv: np.ndarray, x: int, y1: int, y2: int) -> bool:
    """Check interior_v[y1:y2, x] is all True."""
    if y2 <= y1:
        return False
    col = pv[:, x]
    s = col[y2 - 1] - (col[y1 - 1] if y1 > 0 else 0)
    return s == (y2 - y1)


# ----------------------------- Hopcroft–Karp + Kőnig sets ----------------------------- #

def hopcroft_karp(adj: Dict[int, List[int]], n_left: int, n_right: int) -> Tuple[List[int], List[int]]:
    INF = 10**9
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs() -> bool:
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found_free = False
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                u2 = pair_v[v]
                if u2 != -1 and dist[u2] == INF:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
                if u2 == -1:
                    found_free = True
        return found_free

    def dfs(u: int) -> bool:
        for v in adj.get(u, []):
            u2 = pair_v[v]
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                dfs(u)

    return pair_u, pair_v


def konig_reachable_sets(
    adj: Dict[int, List[int]],
    pair_u: List[int],
    pair_v: List[int],
    n_left: int,
    n_right: int,
) -> Tuple[Set[int], Set[int]]:
    """
    ZL/ZR reachable via alternating paths from free left vertices.
    Max Independent Set of chord-nodes = (L ∩ ZL) ∪ (R \ ZR)
    """
    ZL: Set[int] = set()
    ZR: Set[int] = set()

    q = deque()
    for u in range(n_left):
        if pair_u[u] == -1:
            ZL.add(u)
            q.append(("L", u))

    while q:
        side, node = q.popleft()
        if side == "L":
            u = node
            for v in adj.get(u, []):
                if pair_u[u] != v and v not in ZR:
                    ZR.add(v)
                    q.append(("R", v))
        else:
            v = node
            u2 = pair_v[v]
            if u2 != -1 and u2 not in ZL:
                ZL.add(u2)
                q.append(("L", u2))

    return ZL, ZR


# ----------------------------- Level 1: chord candidates (adjacent only) ----------------------------- #

def enumerate_level1_chords_adjacent(mask01: np.ndarray) -> Tuple[List[Chord], List[Chord]]:
    """
    Enumerate candidate chords connecting *adjacent* cogrid concave vertices only.
    This prevents same-orientation overlap and matches the bipartite intersection model.
    """
    conc = concave_vertices(mask01)

    conc_by_y: Dict[int, List[int]] = defaultdict(list)
    conc_by_x: Dict[int, List[int]] = defaultdict(list)
    for x, y in conc:
        conc_by_y[y].append(x)
        conc_by_x[x].append(y)

    interior_h, interior_v = interior_edges(mask01)
    ph, pv = prefix_sums_edges(interior_h, interior_v)

    Hchords: List[Chord] = []
    Vchords: List[Chord] = []

    # Horizontal: only consecutive concaves on same y
    for y, xs in conc_by_y.items():
        xs = sorted(set(xs))
        for x1, x2 in zip(xs, xs[1:]):
            if interior_h.shape[1] == 0:
                continue
            if 0 <= y < interior_h.shape[0] and 0 <= x1 < interior_h.shape[1] and x2 <= interior_h.shape[1]:
                if all_true_h(ph, y, x1, x2):
                    Hchords.append(Chord("H", y, x1, x2))

    # Vertical: only consecutive concaves on same x
    for x, ys in conc_by_x.items():
        ys = sorted(set(ys))
        for y1, y2 in zip(ys, ys[1:]):
            if interior_v.shape[0] == 0:
                continue
            if 0 <= x < interior_v.shape[1] and 0 <= y1 < interior_v.shape[0] and y2 <= interior_v.shape[0]:
                if all_true_v(pv, x, y1, y2):
                    Vchords.append(Chord("V", x, y1, y2))

    return Hchords, Vchords


def build_conflict_graph(Hchords: List[Chord], Vchords: List[Chord]) -> Dict[int, List[int]]:
    """
    Conflict if an H and V chord share any point (intersection or shared endpoint).
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for i, hc in enumerate(Hchords):
        y = hc.fixed
        x1, x2 = hc.a, hc.b
        for j, vc in enumerate(Vchords):
            x = vc.fixed
            y1, y2 = vc.a, vc.b
            if x1 <= x <= x2 and y1 <= y <= y2:
                adj[i].append(j)
    return adj


def select_level1_independent_chords(Hchords: List[Chord], Vchords: List[Chord]) -> List[Chord]:
    """
    Maximum independent set of chord nodes in bipartite conflict graph.
    """
    adj = build_conflict_graph(Hchords, Vchords)
    pair_u, pair_v = hopcroft_karp(adj, len(Hchords), len(Vchords))
    ZL, ZR = konig_reachable_sets(adj, pair_u, pair_v, len(Hchords), len(Vchords))

    chosen: List[Chord] = []
    for i in ZL:
        chosen.append(Hchords[i])
    for j in range(len(Vchords)):
        if j not in ZR:
            chosen.append(Vchords[j])

    return chosen


# ----------------------------- Global cuts + components ----------------------------- #

def add_chord_to_blocks(chord: Chord, block_h: np.ndarray, block_v: np.ndarray) -> None:
    """
    Mark unit edges along the chord as blocked adjacency between pixels.
    block_h[y,x] blocks adjacency across boundary between rows y-1 and y at column x.
    block_v[y,x] blocks adjacency across boundary between cols x-1 and x at row y.
    """
    if chord.orient == "H":
        y = chord.fixed
        block_h[y, chord.a:chord.b] = True
    else:
        x = chord.fixed
        block_v[chord.a:chord.b, x] = True


def components(mask01: np.ndarray, block_h: np.ndarray, block_v: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    4-connected components of inside pixels, where adjacency is blocked by chord cuts.
    """
    H, W = mask01.shape
    inside = mask01.astype(bool)
    seen = np.zeros((H, W), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []

    def neigh(r: int, c: int):
        if r > 0 and inside[r - 1, c] and not block_h[r, c]:
            yield (r - 1, c)
        if r + 1 < H and inside[r + 1, c] and not block_h[r + 1, c]:
            yield (r + 1, c)
        if c > 0 and inside[r, c - 1] and not block_v[r, c]:
            yield (r, c - 1)
        if c + 1 < W and inside[r, c + 1] and not block_v[r, c + 1]:
            yield (r, c + 1)

    for r in range(H):
        for c in range(W):
            if not inside[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nn in neigh(rr, cc):
                    if not seen[nn]:
                        seen[nn] = True
                        q.append(nn)
            comps.append(comp)

    return comps


def component_local_mask(comp: List[Tuple[int, int]], H: int, W: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Local mask for a component within a padded bbox.
    Returns:
      local_mask (uint8 0/1), and (r_off, c_off, Hloc, Wloc)
    """
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    rr0 = max(r0 - 1, 0)
    rr1 = min(r1 + 1, H - 1)
    cc0 = max(c0 - 1, 0)
    cc1 = min(c1 + 1, W - 1)

    Hloc = rr1 - rr0 + 1
    Wloc = cc1 - cc0 + 1
    local = np.zeros((Hloc, Wloc), dtype=np.uint8)
    for r, c in comp:
        local[r - rr0, c - cc0] = 1

    return local, (rr0, cc0, Hloc, Wloc)


# ----------------------------- Level 2: deterministic vertex choice, random direction only ----------------------------- #

def inward_directions_for_concave_vertex(local_mask: np.ndarray, vx: int, vy: int) -> List[Tuple[int, int]]:
    """
    Return inward ray directions among {E,W,S,N} that start with an interior unit edge
    in the *current component*.

    Vertex coordinates here are LOCAL vertex coords in the same frame as local_mask:
      - pixels are (row,col)
      - grid vertices are (x=col, y=row)
    """
    Hloc, Wloc = local_mask.shape

    def interior_h_edge_at(x: int, y: int) -> bool:
        # unit edge (x,y)->(x+1,y) is interior iff pixels above and below exist
        if y <= 0 or y >= Hloc:
            return False
        if x < 0 or x >= Wloc - 1:
            return False
        return (local_mask[y - 1, x] == 1) and (local_mask[y, x] == 1)

    def interior_v_edge_at(x: int, y: int) -> bool:
        # unit edge (x,y)->(x,y+1) is interior iff pixels left and right exist
        if x <= 0 or x >= Wloc:
            return False
        if y < 0 or y >= Hloc - 1:
            return False
        return (local_mask[y, x - 1] == 1) and (local_mask[y, x] == 1)

    dirs: List[Tuple[int, int]] = []
    # E: edge at (vx,vy)
    if interior_h_edge_at(vx, vy):
        dirs.append((+1, 0))
    # W: edge at (vx-1,vy)
    if interior_h_edge_at(vx - 1, vy):
        dirs.append((-1, 0))
    # S: edge at (vx,vy)
    if interior_v_edge_at(vx, vy):
        dirs.append((0, +1))
    # N: edge at (vx,vy-1)
    if interior_v_edge_at(vx, vy - 1):
        dirs.append((0, -1))

    return dirs


def extend_chord(local_mask: np.ndarray, vx: int, vy: int, d: Tuple[int, int]) -> Optional[Chord]:
    """
    Extend a chord from a concave vertex along direction d until boundary,
    staying on interior unit edges of the component.
    Returns a LOCAL chord (in local grid coords), or None.
    """
    Hloc, Wloc = local_mask.shape
    dx, dy = d

    def interior_h_edge_at(x: int, y: int) -> bool:
        if y <= 0 or y >= Hloc:
            return False
        if x < 0 or x >= Wloc - 1:
            return False
        return (local_mask[y - 1, x] == 1) and (local_mask[y, x] == 1)

    def interior_v_edge_at(x: int, y: int) -> bool:
        if x <= 0 or x >= Wloc:
            return False
        if y < 0 or y >= Hloc - 1:
            return False
        return (local_mask[y, x - 1] == 1) and (local_mask[y, x] == 1)

    x, y = vx, vy
    steps = 0

    while True:
        if dx != 0:
            if dx == +1:
                if not interior_h_edge_at(x, y):
                    break
                x += 1
            else:
                if not interior_h_edge_at(x - 1, y):
                    break
                x -= 1
        else:
            if dy == +1:
                if not interior_v_edge_at(x, y):
                    break
                y += 1
            else:
                if not interior_v_edge_at(x, y - 1):
                    break
                y -= 1

        steps += 1
        if steps > (Hloc + Wloc + 10):
            break

    if steps <= 0:
        return None

    if dx != 0:
        y_fixed = vy
        x1, x2 = sorted([vx, x])
        if x2 <= x1:
            return None
        return Chord("H", y_fixed, x1, x2)
    else:
        x_fixed = vx
        y1, y2 = sorted([vy, y])
        if y2 <= y1:
            return None
        return Chord("V", x_fixed, y1, y2)


# ----------------------------- Main GBD decomposition ----------------------------- #

def gbd_decompose(mask01: np.ndarray, seed: int = 0, max_iter: int = 200000) -> List[Tuple[int, int, int, int]]:
    """
    Input: uint8 mask in {0,1}.
    Output: rectangles (row0, col0, height, width).
    """
    uniq = np.unique(mask01)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"Mask must be binary {0,1}; found {uniq.tolist()}")

    H, W = mask01.shape
    rng = random.Random(seed)

    block_h = np.zeros((H + 1, max(W - 1, 0)), dtype=bool)
    block_v = np.zeros((max(H - 1, 0), W + 1), dtype=bool)

    # -------- Level 1: maximal non-intersecting chord set (via bipartite MIS) --------
    Hchords, Vchords = enumerate_level1_chords_adjacent(mask01)
    lvl1_chords = select_level1_independent_chords(Hchords, Vchords)
    for ch in lvl1_chords:
        add_chord_to_blocks(ch, block_h, block_v)

    # -------- Level 2: eliminate concavities deterministically; randomize direction only --------
    # Outer loop: keep splitting until all components have no concave vertices
    it = 0
    while it < max_iter:
        it += 1
        comps = components(mask01, block_h, block_v)

        any_concave_anywhere = False
        did_cut = False

        # deterministic component order (by top-left pixel)
        comps_sorted = sorted(
            comps,
            key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)),
        )

        for comp in comps_sorted:
            local, (r_off, c_off, Hloc, Wloc) = component_local_mask(comp, H, W)

            conc = concave_vertices(local)
            if not conc:
                continue

            any_concave_anywhere = True

            # deterministic vertex choice: pick lexicographically smallest (y,x)
            # (conc returns (x,y))
            conc_sorted = sorted(conc, key=lambda xy: (xy[1], xy[0]))
            vx, vy = conc_sorted[0]

            dirs = inward_directions_for_concave_vertex(local, vx, vy)
            if not dirs:
                # degenerate: cannot cast chord (should be rare in clean rectilinear inputs)
                continue

            # randomize only between the offered inward directions (paper-style)
            d = rng.choice(dirs)
            chord_local = extend_chord(local, vx, vy, d)
            if chord_local is None:
                continue

            # lift chord to global coordinates
            if chord_local.orient == "H":
                chord_global = Chord(
                    "H",
                    fixed=chord_local.fixed + r_off,
                    a=chord_local.a + c_off,
                    b=chord_local.b + c_off,
                )
            else:
                chord_global = Chord(
                    "V",
                    fixed=chord_local.fixed + c_off,
                    a=chord_local.a + r_off,
                    b=chord_local.b + r_off,
                )

            add_chord_to_blocks(chord_global, block_h, block_v)
            did_cut = True
            break  # recompute components after each cut

        if not any_concave_anywhere:
            break
        if any_concave_anywhere and not did_cut:
            # cannot progress further
            break

    # -------- Extract rectangles from final components --------
    rects: List[Tuple[int, int, int, int]] = []
    comps = components(mask01, block_h, block_v)

    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h * w != len(comp):
            raise RuntimeError(
                f"Non-rectangular component detected (bbox area {h*w} != pixels {len(comp)}). "
                f"This usually indicates the input is not a clean rectilinear polygon, "
                f"or the decomposition assumptions are violated."
            )
        rects.append((r0, c0, h, w))

    return rects


# ----------------------------- Visualization ----------------------------- #

def visualize(mask01: np.ndarray, rects: List[Tuple[int, int, int, int]], seed: int = 0, save: Optional[str] = None):
    """
    Left: black/white mask (0=black, 1=white)
    Right: colored rectangles + borders
    """
    H, W = mask01.shape
    rng = random.Random(seed)

    colored = np.zeros((H, W, 3), dtype=float)
    for (r0, c0, h, w) in rects:
        color = (rng.random(), rng.random(), rng.random())
        colored[r0:r0+h, c0:c0+w, :] = color

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title("Binary mask (B/W)")
    axs[0].imshow(mask01, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axs[0].axis("off")

    axs[1].set_title(f"Decomposed into {len(rects)} rectangles")
    axs[1].imshow(colored, interpolation="nearest")
    axs[1].axis("off")

    for (r0, c0, h, w) in rects:
        axs[1].add_patch(Rectangle((c0 - 0.5, r0 - 0.5), w, h, fill=False, linewidth=0))

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


# ----------------------------- Batch runner (directory) ----------------------------- #

def iter_image_files(directory: str):
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"--test-dir must be an existing directory: {directory}")
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


# ----------------------------- CLI ----------------------------- #

def main():
    ap = argparse.ArgumentParser(description="GBD rectangle decomposition for binary images (TIFF supported).")
    ap.add_argument("--input", type=str, default=None, help="Path to a single binary image.")
    ap.add_argument("--test-dir", type=str, default=None, help="Directory of binary test images to iterate.")

    ap.add_argument("--seed", type=int, default=0, help="Random seed (affects only direction choices + colors).")
    ap.add_argument("--max-iter", type=int, default=200000, help="Max iterations for level-2 splitting.")
    ap.add_argument("--save", type=str, default=None, help="Save visualization (single input mode).")
    ap.add_argument("--save-dir", type=str, default=None, help="Save visualizations for directory mode.")
    ap.add_argument("--dump-rects", type=str, default=None, help="Write rectangles JSON (single input mode).")
    ap.add_argument("--dump-summary", type=str, default=None, help="Write per-file JSON summary (dir mode).")
    ap.add_argument("--non-strict", action="store_true",
                    help="Allow non-binary images by auto-binarizing (NOT recommended).")

    args = ap.parse_args()

    if (args.input is None) == (args.test_dir is None):
        raise SystemExit("Provide exactly one of --input OR --test-dir.")

    strict = not args.non_strict

    if args.input:
        mask = load_binary_mask(args.input, strict=strict)
        rects = gbd_decompose(mask, seed=args.seed, max_iter=args.max_iter)

        if args.dump_rects:
            with open(args.dump_rects, "w", encoding="utf-8") as f:
                json.dump([{"r": r, "c": c, "h": h, "w": w} for (r, c, h, w) in rects], f, indent=2)

        visualize(mask, rects, seed=args.seed, save=args.save)
        return

    # directory mode
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    paths = list(iter_image_files(args.test_dir))
    if not paths:
        raise SystemExit(f"No supported image files found in: {args.test_dir}")

    for p in paths:
        rec = {"file": str(p), "ok": False}
        t0 = time.perf_counter()
        try:
            mask = load_binary_mask(str(p), strict=strict)
            rects = gbd_decompose(mask, seed=args.seed, max_iter=args.max_iter)
            rec["rectangles"] = len(rects)
            rec["pixels"] = int(mask.sum())
            rec["ok"] = True

            if save_dir:
                out_path = save_dir / f"{p.stem}_gbd.png"
                visualize(mask, rects, seed=args.seed, save=str(out_path))

        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            rec["seconds"] = time.perf_counter() - t0
            results.append(rec)

    ok = sum(r["ok"] for r in results)
    print(f"Processed {len(results)} file(s). OK: {ok}, Fail: {len(results) - ok}")

    if args.dump_summary:
        with open(args.dump_summary, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
