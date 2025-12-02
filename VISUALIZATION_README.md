# 可视化函数优化说明

## 功能改进

### 1. **连续网格可视化 (Mesh Visualization)**
- 使用 Delaunay 三角剖分构建连续的网格表面
- 比散点图更直观地显示场的连续分布
- 自动从节点坐标生成三角形

### 2. **自动空间电荷区检测 (Automatic Space Charge Detection)**
- 通过统计分析自动检测空间电荷（耗尽）区
- 使用数据分布的百分位数（40%）确定阈值
- 无需手动指定参数

### 3. **耗尽线绘制 (Depletion Boundary)**
- 自动追踪空间电荷区与中性区的边界
- 绘制黑色虚线标记耗尽线位置
- 便于直观观察空间电荷区的形状和范围

### 4. **改进的色彩方案**
- 使用 RdBu_r (Red-Blue Reversed) 色标
  - **红色** = 负值（空间电荷密集区）
  - **白色** = 零值（边界）
  - **蓝色** = 正值（非耗尽区）
- 对称的颜色范围确保零点总是映射到中间（白色）
- TwoSlopeNorm 确保正负值有相等的视觉权重

## 使用方法

### 方法1：使用推理脚本 (infer.py)
```bash
python infer.py --group n43 --sheet 0 --checkpoint outputs/checkpoints/meshgraphnet_epoch_300.pt
```
自动生成：
- `infer_n43_s0_SpaceCharge_pred.png` - 预测场
- `infer_n43_s0_SpaceCharge_true.png` - 真实场
- `infer_n43_s0_SpaceCharge_error.png` - 误差分布

### 方法2：使用高级可视化脚本 (visualize.py)
```bash
# 默认网格模式
python visualize.py --group n43 --sheet 0

# 指定特定检查点
python visualize.py --group n43 --sheet 0 --checkpoint outputs/checkpoints/meshgraphnet_epoch_300.pt

# 使用散点图模式
python visualize.py --group n43 --sheet 0 --mode scatter

# 自定义输出名称
python visualize.py --group n43 --sheet 0 --output-name my_custom_viz
```

## 图像解释

### Prediction (pred.png)
- 模型预测的电位/电场分布
- 红色区域显示强烈的负值（空间电荷区）
- 黑色虚线标记的边界是耗尽线

### Ground Truth (true.png)
- 实际的物理场分布
- 与预测图对比可评估模型精度

### Error (error.png)
- 预测与真实值的差异
- 红色表示预测值偏低，蓝色表示预测值偏高

## 技术细节

### 空间电荷区检测算法
1. 计算字段值的排序分布
2. 取第40百分位值作为阈值（因为大多数空间电荷是负值）
3. 小于阈值的节点标记为耗尽区

### 耗尽线追踪算法
1. 识别跨越耗尽边界的边（一端在耗尽区，一端在非耗尽区）
2. 计算这些边的中点
3. 按圆周角度排序中点（相对于中心）
4. 连接排序后的点形成连续的边界线

## 可视化参数

### scatter_field_comparison() 函数签名
```python
scatter_field_comparison(
    pos: torch.Tensor,                    # [N, 2] 节点坐标
    pred: torch.Tensor,                   # [N] 预测值
    target: torch.Tensor,                 # [N] 真实值
    save_prefix: Path,                    # 输出文件前缀
    title_prefix: str = "",               # 标题前缀
    vmin: Optional[float] = None,         # 颜色范围最小值（自动计算）
    vmax: Optional[float] = None,         # 颜色范围最大值（自动计算）
    edge_index: Optional[torch.Tensor] = None,  # [2, E] 边信息（用于网格）
    use_mesh: bool = True,                # 是否使用网格（True）或散点图（False）
)
```

## 文件列表

- `utils/visualization.py` - 核心可视化函数（已优化）
- `infer.py` - 推理脚本（已更新以使用新可视化）
- `visualize.py` - 独立的高级可视化脚本（新增）
