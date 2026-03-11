"""
multiadam_optimizable.py

MultiAdam 优化器中的可进化核心函数。
对应 PINN 多损失组融合更新问题：如何将各损失组的 Adam 原始更新量
合并为单一参数更新方向，直接决定优化动态和收敛质量。

[优化目标]  在 Burgers1D / Poisson2D_Classic / Heat2D_Multiscale 上降低 L2 相对误差。

[外部知识]
- 各损失组 Adam 原始更新 update_raw[g] = exp_avg[g] / denom[g]，形状与参数相同
- group_weights[g] >= 0，由 ParamScheduler 给出（训练初期通常均匀）
- PINN 常见病态：PDE 残差梯度方向与边界条件梯度方向冲突
- 投影、归一化、动量聚合等非线性融合策略均值得探索
"""

import torch
from torch import Tensor


def compute_group_update(
    exp_avg: Tensor,       # shape: [n_groups, *param_shape]  各组一阶矩
    denom: Tensor,         # shape: [n_groups, *param_shape]  各组 Adam 分母
    group_weights: Tensor, # shape: [n_groups]                各组融合权重
) -> Tensor:
    """
    [MULTIADAM-CORE] 将各损失组的 Adam 原始更新量融合为最终参数更新向量。

    当前实现：线性加权求和（论文默认方案）。
      update_raw[g] = exp_avg[g] / denom[g]
      output        = Σ_g  weight[g] * update_raw[g]

    可探索方向：
      - 梯度投影：去除各组更新方向的冲突分量后再融合
      - 归一化融合：先将 update_raw 单位化再加权
      - 动态软门控：用 softmax(group_weights) 替代直接归一化权重
      - 方差感知：按各组更新量的方差反比调整权重

    Args:
        exp_avg:      各损失组的偏差修正后一阶矩，已 stack 成 [n_groups, *param_shape]
        denom:        各损失组的 Adam 分母（sqrt(二阶矩)/bias_correction + eps）
        group_weights: 各组的融合权重（非负实数，由调度器给出）

    Returns:
        Tensor: 与单个参数形状相同的最终更新向量（后续由外部乘以 step_size）
    """
    update_raw = exp_avg / denom  # [n_groups, *param_shape]
    # 线性加权：广播 group_weights 到参数形状
    w = group_weights.view((-1,) + (1,) * (update_raw.dim() - 1))
    return (update_raw * w).sum(dim=0)
