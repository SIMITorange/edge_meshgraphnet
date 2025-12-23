"""
usage:

:: 1) 放在 dat/grd 同级时（默认input=脚本目录）
python build_meshgraph_hdf5_from_grd_dat.py

:: 2) 指定原始数据目录（你现在的例子）
python build_meshgraph_hdf5_from_grd_dat.py --input "E:\STDB_suhang\Zhangcheng\output\GaN_PNdiode_halfcell_termin_BV_FFLR_3n_5flr_first1_2dif"

:: 3) 如需自定义输出父目录
python build_meshgraph_hdf5_from_grd_dat.py --input "D:\...\1.5int" --output-parent "D:\...\1.5int"
"""
import argparse
import os
import re
import time
from datetime import datetime

import h5py
import numpy as np
from scipy.spatial import Delaunay


TARGET_COLS = [
    "ElectrostaticPotential",
    "eDensity",
    "hDensity",
    "SpaceCharge",
    "ElectricField_x",
    "ElectricField_y",
    "DopingConcentration",
]

DEFAULT_GROUP_REGEX = r"_\d{4}_"

DAT_FILENAME_PATTERN = re.compile(r"Block_n(\d+)_(\d+)_des\.dat$", re.IGNORECASE)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_file_signature(filename: str, group_regex: str) -> str:
    return re.sub(group_regex, "_XXXX_", filename)


def parse_grd_vertices(grd_path: str):
    coordinates = []
    is_parsing_vertices = False
    start_pattern = re.compile(r"Vertices\s*\(\s*(\d+)\s*\)\s*\{")

    with open(grd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if not is_parsing_vertices:
                if start_pattern.search(line):
                    is_parsing_vertices = True
                continue

            if "}" in line:
                content_before = line.split("}")[0].strip()
                if content_before:
                    parts = content_before.split()
                    for i in range(0, len(parts), 2):
                        if i + 1 < len(parts):
                            coordinates.append((float(parts[i]), float(parts[i + 1])))
                break

            parts = line.split()
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    coordinates.append((float(parts[i]), float(parts[i + 1])))

    return np.asarray(coordinates, dtype=np.float32)


def parse_dat_value_count_for_limit(dat_path: str, dataset_name: str = "ElectrostaticPotential"):
    target_found = False
    dataset_pattern = re.compile(rf"Dataset\s*\(\s*\"{re.escape(dataset_name)}\"\s*\)")
    values_pattern = re.compile(r"Values\s*\(\s*(\d+)\s*\)")

    with open(dat_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if not target_found:
                if dataset_pattern.search(line):
                    target_found = True
            else:
                m = values_pattern.search(line)
                if m:
                    return int(m.group(1))
                if "Dataset" in line and "(" in line:
                    target_found = False

    return None


def iter_dat_datasets(dat_path: str):
    with open(dat_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            yield raw.rstrip("\n")


def parse_dat_fields(
    dat_path: str,
    validity_target: str,
    target_cols,
    num_nodes: int,
):
    required_names = set([c.replace("_x", "").replace("_y", "") for c in target_cols])
    out = {k: np.zeros((num_nodes,), dtype=np.float32) for k in target_cols}

    in_info = False
    in_datasets = False
    datasets_list = []

    in_data = False
    in_dataset_block = False

    cur_name = None
    cur_validity = None
    cur_type = None
    cur_dim = None
    cur_value_count = None
    cur_values = []
    in_values = False

    processed_base = set()

    def finalize_current():
        nonlocal cur_name, cur_validity, cur_type, cur_dim, cur_value_count, cur_values, in_dataset_block, in_values
        if not in_dataset_block or cur_name is None:
            return

        base = cur_name
        if base not in required_names:
            pass
        elif base in processed_base:
            pass
        elif datasets_list and base not in datasets_list:
            pass
        elif cur_validity != validity_target:
            pass
        elif cur_type not in ("scalar", "vector"):
            pass
        elif cur_dim is None or cur_value_count is None:
            pass
        else:
            try:
                vals = np.asarray([float(x) for x in cur_values], dtype=np.float32)
            except ValueError:
                vals = None

            if vals is not None and len(vals) == cur_value_count:
                if cur_type == "scalar":
                    if base in target_cols:
                        out[base][:] = vals[:num_nodes]
                        processed_base.add(base)
                elif cur_type == "vector":
                    xk = f"{base}_x"
                    yk = f"{base}_y"
                    if xk in target_cols and yk in target_cols:
                        out[xk][:] = vals[::2][:num_nodes]
                        out[yk][:] = vals[1::2][:num_nodes]
                        processed_base.add(base)

        cur_name = None
        cur_validity = None
        cur_type = None
        cur_dim = None
        cur_value_count = None
        cur_values = []
        in_dataset_block = False
        in_values = False

    for line in iter_dat_datasets(dat_path):
        s = line.strip()
        if not s:
            continue

        if not in_info and re.match(r"Info\s*\{", s):
            in_info = True
            continue
        if in_info and "}" in s:
            in_info = False
            in_datasets = False
            continue

        if in_info:
            if not in_datasets and re.search(r"datasets\s*=\s*\[", s):
                in_datasets = True
            if in_datasets:
                datasets_list += re.findall(r'"([^"]+)"', s)
                if "]" in s:
                    in_datasets = False
            continue

        if not in_data and re.match(r"Data\s*\{", s):
            in_data = True
            continue
        if in_data and s == "}":
            finalize_current()
            in_data = False
            continue

        if not in_data:
            continue

        m = re.search(r'Dataset\s*\(\s*"([^"]+)"\s*\)', s)
        if m:
            finalize_current()
            cur_name = m.group(1)
            in_dataset_block = True
            continue

        if not in_dataset_block or cur_name is None:
            continue

        if s.startswith("validity"):
            vm = re.search(r'\[\s*"([^"]+)"\s*\]', s)
            if vm:
                cur_validity = vm.group(1)
            continue

        if s.startswith("type"):
            cur_type = re.split(r"[=\s]+", s, 1)[1].split("#")[0].strip()
            continue

        if s.startswith("dimension"):
            try:
                cur_dim = int(re.split(r"[=\s]+", s, 1)[1].split("#")[0].strip())
            except Exception:
                cur_dim = None
            continue

        if s.startswith("Values"):
            m2 = re.search(r"\((\d+)\)", s)
            if m2:
                cur_value_count = int(m2.group(1))
                in_values = True
                cur_values = []
            continue

        if in_values:
            if "{" in s:
                s = s.split("{", 1)[1]
            if "}" in s:
                before = s.split("}", 1)[0]
                if before.strip():
                    cur_values.extend(before.split())
                in_values = False
            else:
                cur_values.extend(s.split())

    finalize_current()

    fields = np.stack([out[c] for c in target_cols], axis=1).astype(np.float32)
    return fields


def build_delaunay_edges(coords: np.ndarray) -> np.ndarray:
    tri = Delaunay(coords)
    simplices = tri.simplices

    edges_raw = np.vstack([
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [2, 0]],
    ])
    edges_all = np.vstack([edges_raw, edges_raw[:, [1, 0]]])
    edges_unique = np.unique(edges_all, axis=0)
    mask = edges_unique[:, 0] != edges_unique[:, 1]
    edges_final = edges_unique[mask]
    return edges_final.T.astype(np.int64)


def find_dataset_groups(input_dir: str, group_regex: str):
    """兼容旧逻辑：按 group_regex 去重后只取组首。"""
    grd_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".grd")]

    groups = {}
    for f in grd_files:
        sig = get_file_signature(f, group_regex)
        if sig not in groups:
            groups[sig] = f

    result = []
    for _, grd_file in groups.items():
        base = os.path.splitext(grd_file)[0]
        dat_file = base + ".dat"
        grd_path = os.path.join(input_dir, grd_file)
        dat_path = os.path.join(input_dir, dat_file)
        if os.path.exists(dat_path):
            result.append((grd_path, dat_path, grd_file))

    return sorted(result, key=lambda x: x[2].lower())


def find_steps_grouped_by_n(input_dir: str):
    """对齐旧HDF5语义：同一 n 的不同 step/边界条件全部写入 fields 的 T 维。"""
    files = os.listdir(input_dir)
    dats = [f for f in files if f.lower().endswith(".dat")]

    grouped = {}
    for dat_file in dats:
        m = DAT_FILENAME_PATTERN.match(dat_file)
        if not m:
            continue
        n_id = m.group(1)
        step = int(m.group(2))
        base = os.path.splitext(dat_file)[0]
        grd_file = base + ".grd"
        grd_path = os.path.join(input_dir, grd_file)
        dat_path = os.path.join(input_dir, dat_file)
        if not os.path.exists(grd_path):
            continue
        grouped.setdefault(n_id, []).append((step, grd_path, dat_path))

    for n_id in grouped:
        grouped[n_id].sort(key=lambda x: x[0])

    return grouped


def infer_output_h5(input_parent: str, output_name: str = "meshgraph_data.h5"):
    out_dir = os.path.join(input_parent, "train_data")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, output_name)


def dir_has_block_pairs(dir_path: str) -> bool:
    try:
        for name in os.listdir(dir_path):
            if name.lower().endswith(".dat") and DAT_FILENAME_PATTERN.match(name):
                base = os.path.splitext(name)[0]
                if os.path.exists(os.path.join(dir_path, base + ".grd")):
                    return True
    except Exception:
        return False
    return False


def find_candidate_dataset_dirs(input_path: str):
    """兼容单目录与顶层目录两种输入。

    - 如果 input_path 自身就包含可用 Block_n*_des.dat/.grd 对：返回 [input_path]
    - 否则递归遍历子目录，返回所有包含可用对的目录（按路径排序，保证写入顺序稳定）
    """
    input_path = os.path.abspath(input_path)
    if os.path.isdir(input_path) and dir_has_block_pairs(input_path):
        return [input_path]

    candidate_dirs = []
    for root, dirs, files in os.walk(input_path):
        # 小优化：如果当前root已包含可用对，就认为它是一个数据集目录，不继续深入其子目录
        if dir_has_block_pairs(root):
            candidate_dirs.append(root)
            dirs[:] = []
    return sorted(candidate_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="一体化：从 GRD/DAT 直接生成 MeshGraphNet HDF5 训练数据（无 CSV/XLSX 中间文件）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="原始数据父文件夹（包含 .grd/.dat）。默认=脚本所在目录",
    )
    parser.add_argument(
        "--output-parent",
        type=str,
        default=None,
        help="输出父文件夹。默认=--input（或脚本目录）",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="meshgraph_data.h5",
        help="输出HDF5文件名（保存在 <output_parent>/train_data/ 下）",
    )
    parser.add_argument(
        "--group-regex",
        type=str,
        default=DEFAULT_GROUP_REGEX,
        help="分组去重规则，默认忽略 _0000_ 这段：_\\d{4}_",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["by_n", "legacy_group_first"],
        default="by_n",
        help="by_n: 按 n 聚合全部 step 写入 fields(T, N, F)；legacy_group_first: 兼容旧的去重取组首逻辑",
    )
    parser.add_argument(
        "--validity",
        type=str,
        default="GaNOnGaN_1",
        help="DAT中 validity 目标（与旧脚本一致）",
    )
    parser.add_argument(
        "--no-stack-time",
        action="store_true",
        help="不按“边界条件/步数”堆叠为T维度，只写单帧 fields=(N,F)",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(args.input) if args.input else script_dir
    output_parent = os.path.abspath(args.output_parent) if args.output_parent else input_dir
    output_h5 = infer_output_h5(output_parent, args.output_name)

    print(f"[{_now()}] 输入目录: {input_dir}")
    print(f"[{_now()}] 输出文件: {output_h5}")

    dataset_dirs = find_candidate_dataset_dirs(input_dir)
    if not dataset_dirs:
        raise SystemExit(f"未找到任何包含 Block_n*_des.dat/.grd 的数据目录（输入: {input_dir}）")

    if len(dataset_dirs) == 1:
        print(f"[{_now()}] 发现 1 个数据目录: {dataset_dirs[0]}")
    else:
        print(f"[{_now()}] 发现 {len(dataset_dirs)} 个数据目录，将合并写入同一个HDF5")

    t_all = time.time()
    with h5py.File(output_h5, "w") as h5f:
        h5f.attrs["columns"] = np.array([c.encode("utf8") for c in TARGET_COLS])
        h5f.attrs["method"] = "Delaunay"

        # 全局重编号，避免不同子目录同名n导致混乱
        global_counter = 0

        for d_index, data_dir in enumerate(dataset_dirs, start=1):
            rel = os.path.relpath(data_dir, input_dir)
            print(f"\n[{_now()}] 扫描数据目录 ({d_index}/{len(dataset_dirs)}): {rel}")

            if args.mode == "legacy_group_first":
                # legacy: 每个“组首”单独写一个 nK（不聚合 T 维）
                pairs = find_dataset_groups(data_dir, args.group_regex)
                if not pairs:
                    print(f"  -> [跳过] 未找到可用对: {data_dir}")
                    continue

                for grd_path, dat_path, _ in pairs:
                    t0 = time.time()
                    global_counter += 1
                    group_name = f"n{global_counter}"
                    label = os.path.basename(dat_path)
                    print(f"\n[{_now()}] 写入 {group_name} <- {label}")

                    limit = parse_dat_value_count_for_limit(dat_path, "ElectrostaticPotential")
                    if limit is None:
                        print(f"  -> [跳过] DAT未找到 ElectrostaticPotential Values 计数: {os.path.basename(dat_path)}")
                        continue

                    pos_all = parse_grd_vertices(grd_path)
                    if pos_all.size == 0:
                        print(f"  -> [跳过] GRD未解析到 Vertices: {os.path.basename(grd_path)}")
                        continue

                    if len(pos_all) < limit:
                        print(f"  -> [Warn] GRD节点数({len(pos_all)}) < DAT请求数({limit})，将截断为 {len(pos_all)}")
                        limit = len(pos_all)

                    pos = pos_all[:limit].astype(np.float32)
                    grp = h5f.create_group(group_name)

                    print(f"  -> 节点数: {len(pos)}；构建Delaunay边...", end="", flush=True)
                    t_edge = time.time()
                    edge_index = build_delaunay_edges(pos)
                    print(f" 完成 (E={edge_index.shape[1]}, {time.time()-t_edge:.2f}s)")
                    grp.create_dataset("pos", data=pos, compression="gzip", compression_opts=4)
                    grp.create_dataset("edge_index", data=edge_index, compression="gzip", compression_opts=4)

                    print(f"  -> 解析DAT字段({args.validity})...", end="", flush=True)
                    t_dat = time.time()
                    fields_nf = parse_dat_fields(dat_path, args.validity, TARGET_COLS, num_nodes=len(pos))
                    print(f" 完成 ({time.time()-t_dat:.2f}s)")

                    if args.no_stack_time:
                        grp.create_dataset("fields", data=fields_nf, compression="gzip", compression_opts=4)
                    else:
                        fields = fields_nf[None, :, :]
                        grp.create_dataset("fields", data=fields, maxshape=(None, fields.shape[1], fields.shape[2]), compression="gzip", compression_opts=4)

                    grp.attrs["radius_used"] = -1.0
                    grp.attrs["method"] = "Delaunay"
                    grp.attrs["num_nodes"] = len(pos)
                    grp.attrs["num_edges"] = int(grp["edge_index"].shape[1])
                    grp.attrs["num_sheets"] = int(grp["fields"].shape[0]) if grp["fields"].ndim == 3 else 1
                    grp.attrs["source_dir"] = data_dir
                    grp.attrs["source_dat"] = os.path.basename(dat_path)
                    grp.attrs["source_grd"] = os.path.basename(grd_path)

                    print(f"  -> 写入 {group_name} 完成 (总耗时 {time.time()-t0:.2f}s)")
            else:
                # by_n: 同一 data_dir 内按原始 n_id 聚合所有 step -> fields 的 T 维
                grouped = find_steps_grouped_by_n(data_dir)
                if not grouped:
                    print(f"  -> [跳过] 未找到可用对: {data_dir}")
                    continue

                for n_id, steps in sorted(grouped.items(), key=lambda x: int(x[0])):
                    t_group = time.time()
                    global_counter += 1
                    group_name = f"n{global_counter}"
                    grp = h5f.create_group(group_name)
                    grp.attrs["source_dir"] = data_dir
                    grp.attrs["source_original_n"] = n_id

                    print(f"\n[{_now()}] 写入 {group_name} <- 原始 n{n_id} (step数: {len(steps)})")

                    ds_fields = None
                    num_nodes = None
                    num_edges = None

                    for step, grd_path, dat_path in steps:
                        label = os.path.basename(dat_path)
                        print(f"  -> step {step}: {label}")

                        limit = parse_dat_value_count_for_limit(dat_path, "ElectrostaticPotential")
                        if limit is None:
                            print(f"     [跳过] DAT未找到 ElectrostaticPotential Values 计数")
                            continue

                        pos_all = parse_grd_vertices(grd_path)
                        if pos_all.size == 0:
                            print(f"     [跳过] GRD未解析到 Vertices")
                            continue

                        if len(pos_all) < limit:
                            print(f"     [Warn] GRD节点数({len(pos_all)}) < DAT请求数({limit})，截断为 {len(pos_all)}")
                            limit = len(pos_all)

                        pos = pos_all[:limit].astype(np.float32)

                        if num_nodes is None:
                            num_nodes = len(pos)
                            print(f"     节点数: {num_nodes}；构建Delaunay边...", end="", flush=True)
                            t_edge = time.time()
                            edge_index = build_delaunay_edges(pos)
                            print(f" 完成 (E={edge_index.shape[1]}, {time.time()-t_edge:.2f}s)")
                            grp.create_dataset("pos", data=pos, compression="gzip", compression_opts=4)
                            grp.create_dataset("edge_index", data=edge_index, compression="gzip", compression_opts=4)
                            num_edges = int(edge_index.shape[1])
                        else:
                            if len(pos) != num_nodes:
                                print(f"     [Warn] 节点数变化({len(pos)} != {num_nodes})，跳过该step")
                                continue

                        print(f"     解析DAT字段({args.validity})...", end="", flush=True)
                        t_dat = time.time()
                        fields_nf = parse_dat_fields(dat_path, args.validity, TARGET_COLS, num_nodes=num_nodes)
                        print(f" 完成 ({time.time()-t_dat:.2f}s)")

                        if args.no_stack_time:
                            if ds_fields is None:
                                ds_fields = grp.create_dataset("fields", data=fields_nf, maxshape=(None, fields_nf.shape[0], fields_nf.shape[1]), compression="gzip", compression_opts=4)
                                ds_fields.resize((1, fields_nf.shape[0], fields_nf.shape[1]))
                                ds_fields[0, :, :] = fields_nf
                            else:
                                old_t = ds_fields.shape[0]
                                ds_fields.resize((old_t + 1, ds_fields.shape[1], ds_fields.shape[2]))
                                ds_fields[old_t, :, :] = fields_nf
                        else:
                            frame = fields_nf[None, :, :]
                            if ds_fields is None:
                                ds_fields = grp.create_dataset("fields", data=frame, maxshape=(None, frame.shape[1], frame.shape[2]), compression="gzip", compression_opts=4)
                            else:
                                old_t = ds_fields.shape[0]
                                ds_fields.resize((old_t + 1, ds_fields.shape[1], ds_fields.shape[2]))
                                ds_fields[old_t : old_t + 1, :, :] = frame

                    if num_nodes is None or ds_fields is None:
                        print(f"  -> [Warn] {group_name} 无有效step，删除该group")
                        del h5f[group_name]
                        global_counter -= 1
                        continue

                    grp.attrs["radius_used"] = -1.0
                    grp.attrs["method"] = "Delaunay"
                    grp.attrs["num_nodes"] = int(num_nodes)
                    grp.attrs["num_edges"] = int(num_edges) if num_edges is not None else int(grp["edge_index"].shape[1])
                    grp.attrs["num_sheets"] = int(grp["fields"].shape[0])
                    grp.attrs["source_dat_first"] = os.path.basename(steps[0][2])
                    grp.attrs["source_grd_first"] = os.path.basename(steps[0][1])

                    print(f"  -> 写入 {group_name} 完成 (T={grp['fields'].shape[0]}, 总耗时 {time.time()-t_group:.2f}s)")

    print(f"\n[{_now()}] 全部完成，总耗时 {time.time()-t_all:.2f}s")


if __name__ == "__main__":
    main()
