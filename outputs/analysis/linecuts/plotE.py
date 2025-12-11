import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
from collections import defaultdict

# ==========================================
# 核心逻辑：基于边界梯度的结构敏感度
# ==========================================

def get_boundary_sensitivity(y_coords: np.ndarray, doping_phys: np.ndarray, grad_abs: np.ndarray, window: int = 2):
    """
    【核心修改】只提取 PN 结边界处的梯度。
    物理含义：环宽度的变化 = 边界的移动。
    因此，边界处的梯度直接代表了对【宽度】的敏感度。
    """
    # 1. 排序
    order = np.argsort(y_coords)
    d_sorted = doping_phys[order]
    g_sorted = grad_abs[order]
    
    # 2. 识别符号变化点 (PN 结边界)
    # signs: P区为1, N区为-1
    signs = np.sign(d_sorted)
    # 消除零点噪音
    signs[np.abs(signs) < 0.5] = 0 
    
    # 找到符号跳变的位置 (diff != 0 的地方即为边界)
    diff = np.diff(signs)
    boundary_indices = np.where(diff != 0)[0]
    
    structure_sens = {}
    
    # 3. 遍历每个边界，判断它属于哪个环/间距
    # 逻辑：第1个跳变是 Ring1 的开始，第2个是 Ring1 的结束/Space1 的开始...
    # 我们不仅要看边界，还要把边界归类给“这个边界属于谁的宽度”
    
    # 为了简化，我们遍历区间，取区间两端的边界梯度之和
    current_sign = signs[0]
    if current_sign == 0: # 处理起步
        start_idx = 0
        while start_idx < len(signs) and signs[start_idx] == 0:
            start_idx += 1
        if start_idx < len(signs):
            current_sign = signs[start_idx]
    else:
        start_idx = 0
        
    ring_cnt = 0
    space_cnt = 0
    
    # 重新扫描段
    segments = []
    seg_start = start_idx
    
    for i in range(start_idx + 1, len(signs)):
        if signs[i] != 0 and signs[i] != current_sign:
            # 发现新段，记录上一段
            seg_end = i - 1
            is_ring = (current_sign > 0) # 假设 P 型是 Ring, >0
            
            if is_ring:
                ring_cnt += 1
                label = f"Ring_{ring_cnt}"
            else:
                space_cnt += 1
                label = f"Space_{space_cnt}"
            
            # 【核心计算】：宽度敏感度 = 左边界梯度 + 右边界梯度
            # 我们取边界附近 window 个点的平均值，以防单点噪音
            
            # 左边界梯度 (Left Edge Sensitivity)
            l_idx_start = max(0, seg_start - window)
            l_idx_end = min(len(g_sorted), seg_start + window + 1)
            grad_left = np.mean(g_sorted[l_idx_start : l_idx_end]) if seg_start > 0 else 0
            
            # 右边界梯度 (Right Edge Sensitivity)
            r_idx_start = max(0, seg_end - window)
            r_idx_end = min(len(g_sorted), seg_end + window + 1)
            grad_right = np.mean(g_sorted[r_idx_start : r_idx_end]) if seg_end < len(g_sorted)-1 else 0
            
            # 这一段的总宽度敏感度
            # 物理上：改变宽度通常意味着同时移动左右边界（或单边移动）
            # 这里取最大值或总和，代表“这个区域的边界是否敏感”
            total_edge_sens = (grad_left + grad_right)
            
            structure_sens[label] = total_edge_sens
            
            # 更新状态
            current_sign = signs[i]
            seg_start = i

    # 处理最后一段 (通常是 N-sub 或最后一个 Space)
    # 一般最外层边界对 Ring 设计影响较小，可以选择忽略或记录
    
    return structure_sens

def extract_n_s_from_filename(filename):
    base = os.path.basename(filename)
    match = re.search(r'n(\d+)_s(\d+)', base)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def analyze_file_edges(file_path):
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        grp = f[keys[0]] if len(keys) > 0 else f

        y = np.array(grp['y_coords'])
        doping = np.array(grp['doping'])
        grad_abs = np.array(grp['grad_abs'])
        
        return get_boundary_sensitivity(y, doping, grad_abs)

# ==========================================
# 绘图逻辑
# ==========================================

def generate_region_sort_key(name: str):
    type_score = 0 if "Ring" in name else 1
    match = re.search(r'\d+', name)
    num = int(match.group()) if match else float('inf')
    return (num, type_score)


def aggregate_sensitivities_by_n(file_paths):
    aggregated = defaultdict(lambda: defaultdict(float))
    for f_path in file_paths:
        n_id, _ = extract_n_s_from_filename(f_path)
        if n_id is None:
            continue
        sens_dict = analyze_file_edges(f_path)
        for region, value in sens_dict.items():
            aggregated[n_id][region] += value
    return aggregated


def plot_aggregated_heatmap(aggregated_data):
    if not aggregated_data:
        return

    all_regions = set()
    for region_map in aggregated_data.values():
        all_regions.update(region_map.keys())

    if not all_regions:
        return

    sorted_regions = sorted(all_regions, key=generate_region_sort_key)
    n_labels = sorted(aggregated_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))

    matrix = np.zeros((len(sorted_regions), len(n_labels)))
    for j, n_id in enumerate(n_labels):
        region_map = aggregated_data[n_id]
        for i, region in enumerate(sorted_regions):
            matrix[i, j] = region_map.get(region, 0.0)

    fig_w = max(10, len(n_labels) * 0.6)
    fig_h = max(6, len(sorted_regions) * 0.5)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(matrix, aspect='auto', cmap='RdYlBu_r')

    cbar = plt.colorbar(im)
    cbar.set_label("Sensitivity to Width Change (Edge Gradient)", rotation=270, labelpad=20, fontsize=12)

    plt.xticks(range(len(n_labels)), [f"n{n}" for n in n_labels], rotation=45, ha='right')
    plt.yticks(range(len(sorted_regions)), sorted_regions)

    plt.xlabel("Device Groups (Different n values)", fontsize=12)
    plt.ylabel("Geometric Parameters (Widths)", fontsize=12)
    plt.title("Aggregated Structural Sensitivity Across Geometries", fontsize=14)

    plt.tight_layout()

    output_filename = "width_sensitivity_aggregated.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"-> 汇总热力图生成: {output_filename}")

# ==========================================
# 主程序
# ==========================================

def main():
    files = glob.glob("*.h5")
    if not files:
        print("未找到 .h5 文件")
        return

    aggregated_data = aggregate_sensitivities_by_n(files)
    if not aggregated_data:
        print("未能聚合任何敏感度数据")
        return

    plot_aggregated_heatmap(aggregated_data)

if __name__ == "__main__":
    main()