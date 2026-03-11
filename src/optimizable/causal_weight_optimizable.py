"""
causal_weight_optimizable.py  [ReEvo2D 优化目标]

来源：src/pde/baseclass.py → BaseTimePDE（时序因果损失加权）
描述：时间依赖 PDE 的因果损失权重计算

[背景知识]
对于时间依赖 PDE（如 Burgers、Wave、Heat、NS），标准 PINN 同时优化所有时刻的残差，
忽略了物理上的"因果关系"：t₂ 时刻的解依赖于 t₁ < t₂ 时刻的精确解。

Wang et al. (2022) "Respecting causality is all you need for training
physics-informed neural networks" 提出因果加权方案：
    w(t) = exp(-ε · ∫₀ᵗ L(τ) dτ)

含义：只有当 τ < t 的残差已经足够小时，t 时刻的残差才获得较大权重参与训练。
这强迫网络按时序顺序学习，避免"跨过"困难的早期时刻。

[当前实现]
    compute_causal_weights 默认返回全 1 权重（不启用因果加权），
    与 PINNacle 原始行为完全一致。

[优化空间]
LLM 可以探索：
  - 标准因果权重：w_i = exp(-ε · Σ_{j<i} L_j)，按时间分桶
  - 软因果：平滑的 sigmoid 过渡代替硬截断
  - 自适应 ε：随训练进程逐步增大 ε（从弱约束到强约束）
  - 多尺度因果：粗细两个时间分辨率分别计算权重
  - 反向课程：先训练晚期时刻，再过渡到早期时刻

[调用位置]
  本函数在 BasePDE.use_causal() 中被调用（在 pde_wrapper 内部），
  仅对 time-dependent PDE 有意义（BasePDE / BaseTimePDE）。
  稳态 PDE 中不会调用此函数。

[注意] 请勿修改函数签名（参数名/类型），否则 pde_wrapper 会出错。
"""

import torch
from torch import Tensor


def compute_causal_weights(
    t: Tensor,
    residuals: list,
    epsilon: float = 1.0,
) -> Tensor:
    """
    根据时序因果关系为每个配置点计算损失权重。

    Args:
        t         : 各配置点的时间坐标，shape (N, 1)，值域 [0, T]
        residuals : 各 PDE 损失项的残差列表，每项 shape (N, 1)
                    （由 pde_wrapper 在调用本函数前计算）
        epsilon   : 因果约束强度，越大则对早期时刻的要求越严格

    Returns:
        各配置点的权重，shape (N, 1)
        - 默认（全 1）：不启用因果加权，等同于原始 PINN 训练
        - 因果模式下：早期时刻的损失大 → 晚期时刻的权重小
    """
    return torch.ones_like(t)  # 默认：均匀权重，等同于标准 PINN
