#!/usr/bin/env python3
"""
GBD rectangular decomposition + visualization (binary image inputs only).

Input:
  - A *binary* image file (0/1 or 0/255). Supported: .tif/.tiff/.png/.bmp/.jpg, etc.
    (via Pillow). The script enforces binarity: only two unique values allowed after
    grayscale conversion; otherwise it errors.

Output:
  - Displays original + decomposed rectangles, and optionally saves a figure.

Usage:
  python gbd_binary_visualize.py path/to/mask.tif
  python gbd_binary_visualize.py path/to/mask.png --save out.png
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image  # pillow supports TIFF
import os
from pathlib import Path
import time
import json

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}

def iter_image_files(directory: str):
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"--test-dir must be an existing directory: {directory}")
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p

# ----------------------------- Data structures ----------------------------- #

@dataclass(frozen=True)
class Chord:
    orient: str          # 'H' or 'V'
    fixed: int           # y for H, x for V
    a: int               # start coord along varying axis
    b: int               # end coord along varying axis


# ----------------------------- Binary image I/O ----------------------------- #

def load_binary_mask(path: str, *, strict: bool = True) -> np.ndarray:
    """
    Load a binary image and return a 2D uint8 mask with values in {0,1}.

    strict=True:
      - After converting to grayscale, requires exactly 2 unique values (binary).
      - Otherwise raises ValueError.

    Notes:
      - For TIFF, Pillow reads many variants; this function uses grayscale conversion.
      - If your TIFF has a palette or multi-page, Pillow will read the first frame.
    """
    img = Image.open(path)

    # If multi-page TIFF, take first page/frame
    try:
        img.seek(0)
    except Exception:
        pass

    img = img.convert("L")  # grayscale
    arr = np.array(img)

    # Enforce binary: only two unique values allowed (or one if empty/full object)
    uniq = np.unique(arr)
    if strict:
        if uniq.size not in (1, 2):
            raise ValueError(
                f"Input is not binary: found {uniq.size} unique grayscale values. "
                f"Unique values (first 10): {uniq[:10].tolist()}. "
                f"Please provide a true binary image (e.g., 0/255)."
            )

    # Map to {0,1}: treat the larger value as foreground if there are 2 values.
    if uniq.size == 1:
        # all background or all foreground
        return (arr > 0).astype(np.uint8)
    else:
        lo, hi = int(uniq[0]), int(uniq[-1])
        # background = lo, foreground = hi
        return (arr == hi).astype(np.uint8)


# ----------------------------- Geometry helpers ----------------------------- #

def _concave_vertices(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Concave grid vertices: exactly 3 of the 4 incident pixels are inside."""
    h, w = mask.shape
    p = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode="constant", constant_values=0)
    NW = p[0:h+1, 0:w+1]
    NE = p[0:h+1, 1:w+2]
    SW = p[1:h+2, 0:w+1]
    SE = p[1:h+2, 1:w+2]
    s = NW + NE + SW + SE
    ys, xs = np.where(s == 3)
    return list(zip(xs.tolist(), ys.tolist()))  # (x,y) grid-vertex coords


def _interior_edges(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unit edges fully inside the object (pixel union):
      interior_h[y,x] for edge (x,y)->(x+1,y): mask[y-1,x]==mask[y,x]==1
      interior_v[y,x] for edge (x,y)->(x,y+1): mask[y,x-1]==mask[y,x]==1
    """
    H, W = mask.shape
    mh = mask.astype(bool)

    interior_h = np.zeros((H + 1, max(W - 1, 0)), dtype=bool)
    if H >= 2 and W >= 2:
        interior_h[1:H, :] = mh[0:H-1, 0:W-1] & mh[1:H, 0:W-1]

    interior_v = np.zeros((max(H - 1, 0), W + 1), dtype=bool)
    if H >= 2 and W >= 2:
        interior_v[:, 1:W] = mh[0:H-1, 0:W-1] & mh[0:H-1, 1:W]

    return interior_h, interior_v


def _prefix_sums_edges(interior_h: np.ndarray, interior_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ph = np.cumsum(interior_h.astype(np.int32), axis=1) if interior_h.size else interior_h.astype(np.int32)
    pv = np.cumsum(interior_v.astype(np.int32), axis=0) if interior_v.size else interior_v.astype(np.int32)
    return ph, pv


def _all_true_h(ph: np.ndarray, y: int, x1: int, x2: int) -> bool:
    if x2 <= x1:
        return False
    row = ph[y]
    s = row[x2 - 1] - (row[x1 - 1] if x1 > 0 else 0)
    return s == (x2 - x1)


def _all_true_v(pv: np.ndarray, x: int, y1: int, y2: int) -> bool:
    if y2 <= y1:
        return False
    col = pv[:, x]
    s = col[y2 - 1] - (col[y1 - 1] if y1 > 0 else 0)
    return s == (y2 - y1)


# ----------------------------- Bipartite matching (Hopcroft–Karp) ----------------------------- #

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


def konig_sets(
    adj: Dict[int, List[int]],
    pair_u: List[int],
    pair_v: List[int],
    n_left: int,
    n_right: int,
) -> Tuple[Set[int], Set[int]]:
    """Compute ZL/ZR reachable by alternating paths from free left vertices."""
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


# ----------------------------- Level 1: chords + selection ----------------------------- #

def enumerate_level1_chords(mask: np.ndarray) -> Tuple[List[Chord], List[Chord]]:
    conc = _concave_vertices(mask)
    conc_by_y: Dict[int, List[int]] = defaultdict(list)
    conc_by_x: Dict[int, List[int]] = defaultdict(list)
    for x, y in conc:
        conc_by_y[y].append(x)
        conc_by_x[x].append(y)

    interior_h, interior_v = _interior_edges(mask)
    ph, pv = _prefix_sums_edges(interior_h, interior_v)

    Hchords: List[Chord] = []
    Vchords: List[Chord] = []

    for y, xs in conc_by_y.items():
        xs = sorted(set(xs))
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                if interior_h.shape[1] == 0:
                    continue
                if 0 <= y < interior_h.shape[0] and 0 <= x1 < interior_h.shape[1] and x2 <= interior_h.shape[1]:
                    if _all_true_h(ph, y, x1, x2):
                        Hchords.append(Chord("H", y, x1, x2))

    for x, ys in conc_by_x.items():
        ys = sorted(set(ys))
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                y1, y2 = ys[i], ys[j]
                if interior_v.shape[0] == 0:
                    continue
                if 0 <= x < interior_v.shape[1] and 0 <= y1 < interior_v.shape[0] and y2 <= interior_v.shape[0]:
                    if _all_true_v(pv, x, y1, y2):
                        Vchords.append(Chord("V", x, y1, y2))

    return Hchords, Vchords


def build_conflict_graph(Hchords: List[Chord], Vchords: List[Chord]) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = defaultdict(list)
    for i, hc in enumerate(Hchords):
        y = hc.fixed
        x1, x2 = hc.a, hc.b
        for j, vc in enumerate(Vchords):
            x = vc.fixed
            y1, y2 = vc.a, vc.b
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                adj[i].append(j)
    return adj


def select_level1_independent_chords(Hchords: List[Chord], Vchords: List[Chord]) -> List[Chord]:
    adj = build_conflict_graph(Hchords, Vchords)
    pair_u, pair_v = hopcroft_karp(adj, len(Hchords), len(Vchords))
    ZL, ZR = konig_sets(adj, pair_u, pair_v, len(Hchords), len(Vchords))

    chosen: List[Chord] = []
    for i in ZL:
        chosen.append(Hchords[i])
    for j in range(len(Vchords)):
        if j not in ZR:
            chosen.append(Vchords[j])
    return chosen


# ----------------------------- Cuts & components ----------------------------- #

def _add_chord_to_blocks(chord: Chord, block_h: np.ndarray, block_v: np.ndarray) -> None:
    if chord.orient == "H":
        y = chord.fixed
        block_h[y, chord.a:chord.b] = True
    else:
        x = chord.fixed
        block_v[chord.a:chord.b, x] = True


def _components(mask: np.ndarray, block_h: np.ndarray, block_v: np.ndarray) -> List[List[Tuple[int, int]]]:
    H, W = mask.shape
    inside = mask.astype(bool)
    seen = np.zeros((H, W), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []

    def neighbors(r: int, c: int):
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
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nn in neighbors(rr, cc):
                    if not seen[nn]:
                        seen[nn] = True
                        q.append(nn)
            comps.append(comp)
    return comps


def _component_mask_and_bbox(comp: List[Tuple[int, int]], H: int, W: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    rr0 = max(r0 - 1, 0)
    rr1 = min(r1 + 1, H - 1)
    cc0 = max(c0 - 1, 0)
    cc1 = min(c1 + 1, W - 1)

    lh = rr1 - rr0 + 1
    lw = cc1 - cc0 + 1
    local = np.zeros((lh, lw), dtype=np.uint8)
    for r, c in comp:
        local[r - rr0, c - cc0] = 1
    return local, (rr0, rr1, cc0, cc1)


def _concave_vertices_in_component(local_mask: np.ndarray, off_r: int, off_c: int) -> List[Tuple[int, int]]:
    conc_local = _concave_vertices(local_mask)
    return [(off_c + x, off_r + y) for (x, y) in conc_local]


def _inward_dirs(local_mask: np.ndarray, v: Tuple[int, int], off_r: int, off_c: int,
                block_h: np.ndarray, block_v: np.ndarray) -> List[Tuple[int, int]]:
    Hloc, Wloc = local_mask.shape
    gx, gy = v
    lx = gx - off_c
    ly = gy - off_r

    def interior_h_local(x: int, y: int) -> bool:
        if y <= 0 or y >= Hloc:
            return False
        if x < 0 or x >= Wloc - 1:
            return False
        return (local_mask[y - 1, x] == 1) and (local_mask[y, x] == 1)

    def interior_v_local(x: int, y: int) -> bool:
        if x <= 0 or x >= Wloc:
            return False
        if y < 0 or y >= Hloc - 1:
            return False
        return (local_mask[y, x - 1] == 1) and (local_mask[y, x] == 1)

    dirs = []
    if interior_h_local(lx, ly) and not block_h[gy, gx]:
        dirs.append((+1, 0))      # east
    if interior_h_local(lx - 1, ly) and not block_h[gy, gx - 1]:
        dirs.append((-1, 0))      # west
    if interior_v_local(lx, ly) and not block_v[gy, gx]:
        dirs.append((0, +1))      # south
    if interior_v_local(lx, ly - 1) and not block_v[gy - 1, gx]:
        dirs.append((0, -1))      # north
    return dirs


def _extend_chord(local_mask: np.ndarray, v: Tuple[int, int], d: Tuple[int, int],
                 off_r: int, off_c: int, block_h: np.ndarray, block_v: np.ndarray) -> Optional[Chord]:
    Hloc, Wloc = local_mask.shape
    gx, gy = v
    dx, dy = d
    lx = gx - off_c
    ly = gy - off_r

    def interior_h_local(x: int, y: int) -> bool:
        if y <= 0 or y >= Hloc:
            return False
        if x < 0 or x >= Wloc - 1:
            return False
        return (local_mask[y - 1, x] == 1) and (local_mask[y, x] == 1)

    def interior_v_local(x: int, y: int) -> bool:
        if x <= 0 or x >= Wloc:
            return False
        if y < 0 or y >= Hloc - 1:
            return False
        return (local_mask[y, x - 1] == 1) and (local_mask[y, x] == 1)

    steps = 0
    cur_gx, cur_gy = gx, gy
    cur_lx, cur_ly = lx, ly

    while True:
        if dx != 0:
            if dx == +1:
                if not interior_h_local(cur_lx, cur_ly) or block_h[cur_gy, cur_gx]:
                    break
                cur_gx += 1; cur_lx += 1
            else:
                if not interior_h_local(cur_lx - 1, cur_ly) or block_h[cur_gy, cur_gx - 1]:
                    break
                cur_gx -= 1; cur_lx -= 1
        else:
            if dy == +1:
                if not interior_v_local(cur_lx, cur_ly) or block_v[cur_gy, cur_gx]:
                    break
                cur_gy += 1; cur_ly += 1
            else:
                if not interior_v_local(cur_lx, cur_ly - 1) or block_v[cur_gy - 1, cur_gx]:
                    break
                cur_gy -= 1; cur_ly -= 1

        steps += 1
        if steps > (Hloc + Wloc + 10):
            break

    if steps <= 0:
        return None

    if dx != 0:
        y = gy
        x1, x2 = sorted([gx, cur_gx])
        return Chord("H", y, x1, x2) if x2 > x1 else None
    else:
        x = gx
        y1, y2 = sorted([gy, cur_gy])
        return Chord("V", x, y1, y2) if y2 > y1 else None


# ----------------------------- Main decomposition ----------------------------- #

def gbd_decompose(mask: np.ndarray, seed: int = 0, max_iter: int = 200000) -> List[Tuple[int, int, int, int]]:
    """
    Input mask must be uint8 with values {0,1}.
    Output rectangles as (row0, col0, height, width).
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    uniq = np.unique(mask)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"mask must be binary {0,1}, found values: {uniq.tolist()}")

    H, W = mask.shape
    rng = random.Random(seed)

    block_h = np.zeros((H + 1, max(W - 1, 0)), dtype=bool)
    block_v = np.zeros((max(H - 1, 0), W + 1), dtype=bool)

    # Level 1
    Hchords, Vchords = enumerate_level1_chords(mask)
    for ch in select_level1_independent_chords(Hchords, Vchords):
        _add_chord_to_blocks(ch, block_h, block_v)

    # Level 2
    for _ in range(max_iter):
        comps = _components(mask, block_h, block_v)
        any_concave = False
        made_cut = False

        for comp in comps:
            local, (rr0, rr1, cc0, cc1) = _component_mask_and_bbox(comp, H, W)
            conc = _concave_vertices_in_component(local, rr0, cc0)
            if not conc:
                continue

            any_concave = True
            v = conc[0]
            dirs = _inward_dirs(local, v, rr0, cc0, block_h, block_v)
            if not dirs:
                continue

            chord = _extend_chord(local, v, dirs[0], rr0, cc0, block_h, block_v)
            if chord is None:
                continue

            _add_chord_to_blocks(chord, block_h, block_v)
            made_cut = True
            break

        if not any_concave:
            break
        if any_concave and not made_cut:
            break

    # Rectangles from final components
    rects: List[Tuple[int, int, int, int]] = []
    comps = _components(mask, block_h, block_v)
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
                f"Input must be a clean rectilinear object; try another seed or inspect the mask."
            )
        rects.append((r0, c0, h, w))

    return rects


# ----------------------------- Visualization ----------------------------- #

def visualize(mask: np.ndarray, rects: List[Tuple[int, int, int, int]], seed: int = 0, save: Optional[str] = None):
    """
    Show:
      left: binary mask
      right: rectangle tiling (filled random colors) + borders
    """
    H, W = mask.shape
    rng = random.Random(seed)

    colored = np.zeros((H, W, 3), dtype=float)
    for (r0, c0, h, w) in rects:
        color = (rng.random(), rng.random(), rng.random())
        colored[r0:r0+h, c0:c0+w, :] = color

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_title("Binary mask")
    axs[0].imshow(mask, interpolation="nearest", cmap="gray", vmin=0, vmax=1)
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


# ----------------------------- CLI ----------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="GBD decomposition + visualization for binary images."
    )

    # either a single file OR a directory
    ap.add_argument("--input", type=str, default=None, help="Path to a single binary image.")
    ap.add_argument("--test-dir", type=str, default=None,
                    help="Directory containing binary test images. Iterates all supported files.")

    ap.add_argument("--seed", type=int, default=0, help="Seed for level-2 choices and colors.")
    ap.add_argument("--max-iter", type=int, default=200000, help="Max iterations for level-2 splitting.")
    ap.add_argument("--save-dir", type=str, default=None,
                    help="If set, saves a visualization image per input into this directory.")
    ap.add_argument("--dump-summary", type=str, default=None,
                    help="If set, writes a JSON summary (rect counts, timings, failures).")
    ap.add_argument("--non-strict", action="store_true",
                    help="Allow non-binary images by auto-binarizing (NOT recommended).")

    args = ap.parse_args()

    if (args.input is None) == (args.test_dir is None):
        raise SystemExit("Provide exactly one of --input OR --test-dir")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.input:
        paths = [Path(args.input)]
    else:
        paths = list(iter_image_files(args.test_dir))
        if not paths:
            raise SystemExit(f"No supported image files found in: {args.test_dir}")

    for p in paths:
        rec = {"file": str(p), "ok": False}
        t0 = time.perf_counter()
        try:
            print(f"Processing file: {p}")
            mask = load_binary_mask(str(p), strict=not args.non_strict)
            rects = gbd_decompose(mask, seed=args.seed, max_iter=args.max_iter)
            rec["rectangles"] = len(rects)
            rec["pixels"] = int(mask.sum())
            rec["ok"] = True

            # save visualization if requested
            if save_dir:
                out_path = save_dir / f"{p.stem}_gbd.png"
                visualize(mask, rects, seed=args.seed, save=str(out_path))
            else:
                visualize(mask, rects, seed=args.seed, save=None)

        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            rec["seconds"] = time.perf_counter() - t0
            results.append(rec)

    # print a small console summary
    ok = sum(r["ok"] for r in results)
    print(f"Processed {len(results)} file(s). OK: {ok}, Fail: {len(results)-ok}")
    if args.test_dir:
        # show worst/slowest few
        slowest = sorted(results, key=lambda r: r["seconds"], reverse=True)[:5]
        print("Slowest:")
        for r in slowest:
            print(f"  {r['seconds']:.3f}s  ok={r['ok']}  rects={r.get('rectangles')}  {r['file']}")

    if args.dump_summary:
        with open(args.dump_summary, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()