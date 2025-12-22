"""\
Generate perturbed graphs by scanning *global* ring widths and *global* space widths.

This is a variant of `generate_perturbed_graphs.py`.

What changes
- Instead of perturbing one segment at a time, we apply a *uniform* width to:
  - all P-type segments (Ring_*) -> `ring_width`
  - all N-type segments (Space_*) -> `space_width`
- We scan a grid:
  - ring_width: [2.0, 5.0] with step 0.5
  - space_width: [0.5, 5.0] with step 0.5
- For each (ring_width, space_width), generate exactly ONE graph.

Outputs
- Graphs saved to `outputs/generated_graphs_global_scan/` by default.
- Each graph is stored as `.npz` with keys:
  pos, edge_index, doping, vds, meta

Usage
  python generate_global_width_space_scan_graphs.py --group n43 --sheet 0 --edge_method tri

Notes
- The segmentation is derived from doping sign along y-axis (same as original generator).
- Doping is kept fixed (copied from base sample), only y coordinates are warped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib.tri as mtri

import config


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    sign: int  # +1 for P (Ring), -1 for N (Space)

    @property
    def width(self) -> float:
        return float(self.end - self.start)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global scan generator: ring_width x space_width")
    p.add_argument("--group", type=str, required=True, help="Base group name, e.g. n43")
    p.add_argument("--sheet", type=int, required=True, help="Base sheet index")
    p.add_argument("--out_dir", type=str, default=None, help="Output folder")

    p.add_argument("--ring_min", type=float, default=2.0)
    p.add_argument("--ring_max", type=float, default=5.0)
    p.add_argument("--ring_step", type=float, default=0.5)

    p.add_argument("--space_min", type=float, default=0.5)
    p.add_argument("--space_max", type=float, default=5.0)
    p.add_argument("--space_step", type=float, default=0.5)

    p.add_argument("--min_segment_frac", type=float, default=0.005, help="Drop tiny segments below this fraction")
    p.add_argument("--y_bins", type=int, default=512)
    p.add_argument(
        "--edge_method",
        type=str,
        default="tri",
        choices=["tri", "knn"],
        help="How to build edges: tri (Delaunay triangulation) or knn (kNN via KDTree)",
    )
    p.add_argument("--k_nn", type=int, default=8)
    return p.parse_args()


def _robust_sign_from_doping(doping: np.ndarray, eps: float = 0.0) -> np.ndarray:
    s = np.sign(doping)
    if eps > 0:
        s[np.abs(doping) < eps] = 0
    return s


def extract_segments_from_base(pos: np.ndarray, doping: np.ndarray, y_bins: int, min_segment_frac: float) -> List[Segment]:
    y = pos[:, 1]
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_max <= y_min:
        raise ValueError("Invalid y range")

    edges = np.linspace(y_min, y_max, y_bins + 1)
    bin_idx = np.clip(np.digitize(y, edges) - 1, 0, y_bins - 1)

    signs = np.zeros(y_bins, dtype=np.int32)
    for b in range(y_bins):
        mask = bin_idx == b
        if not np.any(mask):
            signs[b] = 0
            continue
        s = _robust_sign_from_doping(doping[mask])
        s_nz = s[s != 0]
        if s_nz.size == 0:
            signs[b] = 0
        else:
            signs[b] = 1 if np.sum(s_nz > 0) >= np.sum(s_nz < 0) else -1

    for i in range(1, y_bins):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    for i in range(y_bins - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]

    segments: List[Segment] = []
    cur = int(signs[0])
    seg_start_bin = 0
    for i in range(1, y_bins):
        if int(signs[i]) != cur:
            seg_end_bin = i - 1
            start = float(edges[seg_start_bin])
            end = float(edges[seg_end_bin + 1])
            if cur != 0:
                segments.append(Segment(start=start, end=end, sign=cur))
            cur = int(signs[i])
            seg_start_bin = i

    start = float(edges[seg_start_bin])
    end = float(edges[-1])
    if cur != 0:
        segments.append(Segment(start=start, end=end, sign=cur))

    y_range = y_max - y_min
    keep: List[Segment] = []
    for seg in segments:
        if seg.width / y_range >= min_segment_frac:
            keep.append(seg)

    if len(keep) < 2:
        raise ValueError("Too few segments found; try increasing y_bins or lowering min_segment_frac")

    merged: List[Segment] = []
    for seg in keep:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if prev.sign == seg.sign:
            merged[-1] = Segment(start=prev.start, end=seg.end, sign=prev.sign)
        else:
            merged.append(seg)

    return merged


def label_parameters(segments: List[Segment]) -> List[str]:
    labels: List[str] = []
    ring_cnt = 0
    space_cnt = 0
    for seg in segments:
        if seg.sign > 0:
            ring_cnt += 1
            labels.append(f"Ring_{ring_cnt}")
        else:
            space_cnt += 1
            labels.append(f"Space_{space_cnt}")
    return labels


def build_warp_mapping(segments: List[Segment], new_widths: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    if len(segments) != len(new_widths):
        raise ValueError("segments and new_widths mismatch")

    y0 = segments[0].start
    old_breaks = [segments[0].start]
    for seg in segments:
        old_breaks.append(seg.end)
    old_breaks = np.asarray(old_breaks, dtype=np.float64)

    new_breaks = [y0]
    cur = y0
    for w in new_widths:
        cur += float(w)
        new_breaks.append(cur)
    new_breaks = np.asarray(new_breaks, dtype=np.float64)

    return old_breaks, new_breaks


def warp_y(y: np.ndarray, old_breaks: np.ndarray, new_breaks: np.ndarray) -> np.ndarray:
    return np.interp(y, old_breaks, new_breaks).astype(np.float32)


def _unique_undirected_edges(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    src_all = np.concatenate([src, dst], axis=0)
    dst_all = np.concatenate([dst, src], axis=0)

    key = src_all * int(num_nodes) + dst_all
    order = np.argsort(key)
    key_sorted = key[order]
    unique_mask = np.ones_like(key_sorted, dtype=bool)
    unique_mask[1:] = key_sorted[1:] != key_sorted[:-1]

    src_u = src_all[order][unique_mask]
    dst_u = dst_all[order][unique_mask]
    return np.stack([src_u, dst_u], axis=0)


def triangulation_edges(pos: np.ndarray) -> np.ndarray:
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tris = tri.triangles
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0).astype(np.int64)

    src = edges[:, 0]
    dst = edges[:, 1]
    edge_index = np.stack([src, dst], axis=0)
    return _unique_undirected_edges(edge_index, num_nodes=pos.shape[0])


def knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("edge_method=knn requires scipy (scipy.spatial.cKDTree)") from e

    pts = pos.astype(np.float64)
    N = pts.shape[0]
    kk = int(min(max(k, 1), N - 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=kk + 1)
    nn = idx[:, 1:]

    src = np.repeat(np.arange(N, dtype=np.int64), nn.shape[1])
    dst = nn.reshape(-1).astype(np.int64)
    edge_index = np.stack([src, dst], axis=0)
    return _unique_undirected_edges(edge_index, num_nodes=N)


def load_base_sample(group: str, sheet: int) -> Tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(str(config.HDF5_PATH), "r") as f:
        grp = f[group]
        pos = grp["pos"][:].astype(np.float32)
        fields = grp["fields"][sheet].astype(np.float32)

    doping = fields[:, config.FIELD_TO_INDEX["DopingConcentration"]].astype(np.float32)
    vds = float(fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max())
    return pos, doping, vds


def _frange_inclusive(vmin: float, vmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((vmax - vmin) / step + 1e-9))
    vals = [float(vmin + i * step) for i in range(n + 1)]
    if vals and vals[-1] < vmax - 1e-9:
        vals.append(float(vmax))
    # clamp tiny floating noise
    return [round(v, 6) for v in vals if v <= vmax + 1e-9]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else (config.OUTPUT_DIR / "generated_graphs_global_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pos, base_doping, base_vds = load_base_sample(args.group, args.sheet)
    segments = extract_segments_from_base(
        pos=base_pos,
        doping=base_doping,
        y_bins=int(args.y_bins),
        min_segment_frac=float(args.min_segment_frac),
    )
    labels = label_parameters(segments)

    ring_vals = _frange_inclusive(float(args.ring_min), float(args.ring_max), float(args.ring_step))
    space_vals = _frange_inclusive(float(args.space_min), float(args.space_max), float(args.space_step))

    # Save segmentation summary for traceability
    np.savez(
        out_dir / f"{args.group}_s{args.sheet}_base_segments.npz",
        labels=np.asarray(labels, dtype=object),
        widths=np.asarray([s.width for s in segments], dtype=np.float32),
        signs=np.asarray([s.sign for s in segments], dtype=np.int32),
        group=args.group,
        sheet=args.sheet,
        ring_vals=np.asarray(ring_vals, dtype=np.float32),
        space_vals=np.asarray(space_vals, dtype=np.float32),
    )

    total = 0
    for ring_w in ring_vals:
        for space_w in space_vals:
            new_widths: List[float] = []
            for seg in segments:
                if seg.sign > 0:
                    new_widths.append(float(ring_w))
                else:
                    new_widths.append(float(space_w))

            old_breaks, new_breaks = build_warp_mapping(segments, new_widths)
            new_pos = base_pos.copy()
            new_pos[:, 1] = warp_y(base_pos[:, 1], old_breaks, new_breaks)

            if args.edge_method == "tri":
                edge_index = triangulation_edges(new_pos)
            else:
                edge_index = knn_edges(new_pos, k=int(args.k_nn))

            meta: Dict = {
                "base_group": args.group,
                "base_sheet": int(args.sheet),
                "scan": "global_ring_space",
                "ring_width": float(ring_w),
                "space_width": float(space_w),
                "ring_min": float(args.ring_min),
                "ring_max": float(args.ring_max),
                "ring_step": float(args.ring_step),
                "space_min": float(args.space_min),
                "space_max": float(args.space_max),
                "space_step": float(args.space_step),
                "labels": labels,
            }

            fname = f"{args.group}_s{args.sheet}_scan_ring{ring_w:.2f}_space{space_w:.2f}.npz"
            np.savez(
                out_dir / fname,
                pos=new_pos.astype(np.float32),
                edge_index=edge_index.astype(np.int64),
                doping=base_doping.astype(np.float32),
                vds=np.asarray([base_vds], dtype=np.float32),
                meta=meta,
            )
            total += 1

    print(f"Generated {total} graphs into {out_dir}")


if __name__ == "__main__":
    main()
