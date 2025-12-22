"""\
Generate perturbed FLR (field limiting ring) geometry graphs from a base HDF5 sample.

What this script does
- Reads a base sample (e.g. --group n43 --sheet 0) from the training HDF5.
- Derives a 1D segmentation along the y-axis from the doping sign (P/N) similar
  to the idea used in outputs/analysis/linecuts/plotE.py.
- Interprets consecutive sign-stable segments as alternating Ring (P) / Space (N)
  regions and extracts their widths in y.
- Creates perturbed variants by changing ONE parameter at a time by +/-1 (um in
  y-units), generating TWO variants for + and TWO variants for -.
- Rebuilds node positions by applying a piecewise-linear warp of y coordinates
  that realizes the new segment widths, while keeping x unchanged.
- Keeps doping fixed to the base sample's doping (as requested), and keeps depth
  implicitly fixed (2D graph, pos[:,0],pos[:,1]).
- Re-triangulates the new point cloud to obtain new edges.
- Saves each new graph as an .npz that can be loaded into an inference script.

Saved format (per graph)
- pos: float32 [N,2] (x,y)
- edge_index: int64 [2,E]
- doping: float32 [N]
- vds: float32 scalar
- meta: dict with base/perturb info (saved via np.savez allow_pickle)

Note
- This is a geometry-warp approach: it reuses the base mesh nodes and doping,
  only moving y-coordinates to reflect width/spacing changes.
- If your y units are not micrometers, treat --delta as "1 unit".

Useage:
    python generate_perturbed_graphs.py --group n43 --sheet 0 --delta 1 --replicates 2 --edge_method tri
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
    sign: int  # +1 for P, -1 for N

    @property
    def width(self) -> float:
        return float(self.end - self.start)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate perturbed graphs from a base HDF5 sample")
    p.add_argument("--group", type=str, required=True, help="Base group name, e.g. n43")
    p.add_argument("--sheet", type=int, required=True, help="Base sheet index")
    p.add_argument("--out_dir", type=str, default=None, help="Output folder (default outputs/generated_graphs)")
    p.add_argument("--delta", type=float, default=1.0, help="Magnitude of width/spacing change")
    p.add_argument("--replicates", type=int, default=2, help="How many variants per sign (+/-) per parameter")
    p.add_argument("--min_segment_frac", type=float, default=0.005, help="Drop tiny segments below this fraction of y-range")
    p.add_argument("--y_bins", type=int, default=512, help="Bins used to build a stable 1D sign profile")
    p.add_argument(
        "--edge_method",
        type=str,
        default="tri",
        choices=["tri", "knn"],
        help="How to build edges: tri (Delaunay triangulation) or knn (kNN via KDTree)",
    )
    p.add_argument("--k_nn", type=int, default=8, help="kNN used to form undirected edges (edge_method=knn)")
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

    # Bin y and take a robust sign per bin.
    edges = np.linspace(y_min, y_max, y_bins + 1)
    bin_idx = np.clip(np.digitize(y, edges) - 1, 0, y_bins - 1)

    signs = np.zeros(y_bins, dtype=np.int32)
    for b in range(y_bins):
        mask = bin_idx == b
        if not np.any(mask):
            signs[b] = 0
            continue
        s = _robust_sign_from_doping(doping[mask])
        # majority vote ignoring zeros
        s_nz = s[s != 0]
        if s_nz.size == 0:
            signs[b] = 0
        else:
            signs[b] = 1 if np.sum(s_nz > 0) >= np.sum(s_nz < 0) else -1

    # Forward-fill / backward-fill zeros to reduce noise
    for i in range(1, y_bins):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    for i in range(y_bins - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]

    # Extract contiguous segments
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
    # last
    start = float(edges[seg_start_bin])
    end = float(edges[-1])
    if cur != 0:
        segments.append(Segment(start=start, end=end, sign=cur))

    # Filter tiny segments
    y_range = y_max - y_min
    keep: List[Segment] = []
    for seg in segments:
        if seg.width / y_range >= min_segment_frac:
            keep.append(seg)

    if len(keep) < 2:
        raise ValueError("Too few segments found; try increasing y_bins or lowering min_segment_frac")

    # Merge adjacent same-sign segments (after filtering)
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
    # piecewise-linear mapping defined by breakpoints
    return np.interp(y, old_breaks, new_breaks).astype(np.float32)


def _unique_undirected_edges(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    # remove self-loops
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    # add reverse to ensure undirected
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
    # Delaunay triangulation via matplotlib (no huge memory blow-up)
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tris = tri.triangles  # (T, 3)
    a = tris[:, [0, 1]].reshape(-1)
    b = tris[:, [1, 2]].reshape(-1)
    c = tris[:, [2, 0]].reshape(-1)
    src = np.concatenate([a[0::2], b[0::2], c[0::2]])
    dst = np.concatenate([a[1::2], b[1::2], c[1::2]])
    edges = np.stack([src, dst], axis=0).astype(np.int64)
    return _unique_undirected_edges(edges, num_nodes=pos.shape[0])


def knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    # Scalable kNN using KDTree (requires scipy). Kept as an option.
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("edge_method=knn requires scipy (scipy.spatial.cKDTree)") from e

    pts = pos.astype(np.float64)
    N = pts.shape[0]
    kk = int(min(max(k, 1), N - 1))
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=kk + 1)  # includes self at index 0
    nn = idx[:, 1:]

    src = np.repeat(np.arange(N, dtype=np.int64), nn.shape[1])
    dst = nn.reshape(-1).astype(np.int64)
    edges = np.stack([src, dst], axis=0)
    return _unique_undirected_edges(edges, num_nodes=N)


def load_base_sample(group: str, sheet: int) -> Tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(str(config.HDF5_PATH), "r") as f:
        grp = f[group]
        pos = grp["pos"][:].astype(np.float32)
        fields = grp["fields"][sheet].astype(np.float32)

    doping = fields[:, config.FIELD_TO_INDEX["DopingConcentration"]].astype(np.float32)
    vds = float(fields[:, config.FIELD_TO_INDEX["ElectrostaticPotential"]].max())
    return pos, doping, vds


def build_variants(base_segments: List[Segment], labels: List[str], delta: float, replicates: int) -> List[Dict]:
    widths = [s.width for s in base_segments]
    variants: List[Dict] = []
    for i, name in enumerate(labels):
        for sign, tag in [(+1, "plus"), (-1, "minus")]:
            for r in range(replicates):
                new_widths = widths.copy()
                new_widths[i] = max(1e-6, float(new_widths[i] + sign * delta))
                variants.append(
                    {
                        "param": name,
                        "direction": tag,
                        "replicate": r,
                        "new_widths": new_widths,
                    }
                )
    return variants


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else (config.OUTPUT_DIR / "generated_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pos, base_doping, base_vds = load_base_sample(args.group, args.sheet)

    segments = extract_segments_from_base(
        pos=base_pos,
        doping=base_doping,
        y_bins=args.y_bins,
        min_segment_frac=args.min_segment_frac,
    )
    labels = label_parameters(segments)

    # Save extracted base parameters for traceability
    base_params = {labels[i]: segments[i].width for i in range(len(segments))}
    np.savez(
        out_dir / f"{args.group}_s{args.sheet}_base_params.npz",
        labels=np.asarray(labels, dtype=object),
        widths=np.asarray([s.width for s in segments], dtype=np.float32),
        signs=np.asarray([s.sign for s in segments], dtype=np.int32),
        base_params=base_params,
        group=args.group,
        sheet=args.sheet,
    )

    variants = build_variants(segments, labels, delta=float(args.delta), replicates=int(args.replicates))

    # Precompute warp for each variant and save graph
    for v in variants:
        old_breaks, new_breaks = build_warp_mapping(segments, v["new_widths"])
        new_pos = base_pos.copy()
        new_pos[:, 1] = warp_y(base_pos[:, 1], old_breaks, new_breaks)

        if args.edge_method == "tri":
            edge_index = triangulation_edges(new_pos)
        else:
            edge_index = knn_edges(new_pos, k=int(args.k_nn))

        meta = {
            "base_group": args.group,
            "base_sheet": int(args.sheet),
            "param": v["param"],
            "direction": v["direction"],
            "replicate": int(v["replicate"]),
            "delta": float(args.delta),
            "labels": labels,
        }

        fname = f"{args.group}_s{args.sheet}_perturb_{v['param']}_{v['direction']}_r{v['replicate']}.npz"
        np.savez(
            out_dir / fname,
            pos=new_pos.astype(np.float32),
            edge_index=edge_index.astype(np.int64),
            doping=base_doping.astype(np.float32),
            vds=np.asarray([base_vds], dtype=np.float32),
            meta=meta,
        )

    print(f"Generated {len(variants)} graphs into {out_dir}")


if __name__ == "__main__":
    main()
