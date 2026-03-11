"""
lra_optimizable.py

学习率退火（Learning Rate Annealing, LRA）中边界条件损失权重的自适应更新规则。
决定了每一步如何根据梯度信息调整各边界条件损失的相对权重。

[优化目标]  加速 PINN 对边界条件的满足，降低最终 L2 相对误差。

[外部知识]
- m_grad_r: PDE 残差损失对全参数梯度的最大绝对值（标量，LRA 论文使用 max-norm）
- grad_mean_i: 第 i 个 BC 损失对全参数梯度的平均绝对值（标量）
- current_weight_i: 第 i 个 BC 损失的当前权重
- alpha: EMA 平滑系数（默认 0.1，越大响应越快但越不稳定）
- LRA 论文（PINN-LRA）核心思路：使 max|∇L_pde| ≈ mean|∇L_bc_i| * λ_i
- 权重更新使用 EMA 而非硬替换，以避免训练震荡
"""


def adapt_boundary_weight(
    m_grad_r: float,          # PDE 梯度最大绝对值（全参数的 max-norm）
    grad_mean_i: float,       # 第 i 个 BC 损失的梯度均值绝对值
    current_weight_i: float,  # 第 i 个 BC 损失的当前权重
    alpha: float,             # EMA 平滑系数 ∈ (0, 1]
) -> float:
    """
    [LRA-UPDATE] 计算第 i 个边界条件损失的新权重（EMA 更新）。

    当前实现：LRA 论文原始公式（指数移动平均）。
      λ_hat = m_grad_r / (grad_mean_i * current_weight_i + eps)
      λ_new  = (1 - alpha) * current_weight_i + alpha * λ_hat

    可探索方向：
      - 自适应 alpha：残差大时 alpha 大（快速响应），残差小时 alpha 小（稳定）
      - 权重裁剪：将 λ_new 限制在 [λ_min, λ_max] 区间防止爆炸
      - 二阶更新：用梯度的二阶矩信息（类 Adam）计算 λ_hat
      - 相对变化限制：|λ_new - λ_old| / λ_old ≤ δ（Trust Region 风格）
      - 符号感知：当 PDE 和 BC 梯度方向冲突时增加 BC 权重更积极

    Args:
        m_grad_r:        PDE 残差损失梯度的最大绝对值
        grad_mean_i:     第 i 个 BC 损失梯度的平均绝对值
        current_weight_i: 当前该 BC 的权重（用于计算 EMA 和归一化）
        alpha:           EMA 平滑系数

    Returns:
        float: 更新后的第 i 个 BC 损失权重
    """
    eps = 1e-8
    lambda_hat = m_grad_r / (grad_mean_i * current_weight_i + eps)
    return (1.0 - alpha) * current_weight_i + alpha * lambda_hat
