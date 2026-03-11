"""
ntk_optimizable.py

基于神经切线核（NTK）理论的损失权重自适应函数。
决定了 PINN 中 PDE 残差损失与边界条件损失的动态再平衡策略。

[优化目标]  降低训练过程中梯度病态问题，加快收敛，降低 L2 相对误差。

[外部知识]
- m_grad_r: PDE 残差损失的梯度 L2 模（标量，全参数聚合）
- m_grad_b: 所有边界条件损失之和的梯度 L2 模（标量）
- NTK 原论文建议权重使 λ_pde * ||∇L_pde|| ≈ λ_bc * ||∇L_bc||
- 当 m_grad_r >> m_grad_b 时，边界条件主导训练，需降低 BC 权重
- 平滑更新（EMA）比硬替换更稳定，但响应较慢
"""

import math


def adapt_loss_weights(
    m_grad_r: float,   # PDE 残差梯度模的平方（sum of squared abs grads）
    m_grad_b: float,   # 边界条件梯度模的平方
    num_pde: int,      # PDE 损失分量数量
    num_bc: int,       # 边界条件损失分量数量
    current_weights: list, # 当前损失权重列表（长度 = num_pde + num_bc）
    iteration: int,    # 当前训练迭代步（从 1 开始）
) -> list:
    """
    [NTK-ADAPT] 根据 PDE 梯度模与边界条件梯度模动态调整各损失分量的权重。

    当前实现：NTK 论文原始公式（硬替换，无平滑）。
      λ_pde = (m_grad_r + m_grad_b) / m_grad_r
      λ_bc  = (m_grad_r + m_grad_b) / m_grad_b

    可探索方向：
      - 指数移动平均（EMA）平滑：λ_new = α*λ_hat + (1-α)*λ_old
      - 对数尺度权重：避免梯度模比值过大时权重爆炸
      - 分层适配：对不同类型边界条件（Dirichlet/Neumann）分别计算
      - 周期性更新：每 K 步更新一次，降低计算开销

    Args:
        m_grad_r:        PDE 梯度模的平方和（来自 abs(p.grad) 的 L2 平方）
        m_grad_b:        BC 梯度模的平方和
        num_pde:         PDE 损失分量数
        num_bc:          BC 损失分量数
        current_weights: 当前权重列表（会被本函数更新并返回）
        iteration:       当前迭代步数（可用于实现 warmup 等策略）

    Returns:
        list: 更新后的权重列表（与 current_weights 等长）
    """
    eps = 1e-8  # 防止除以零
    total = m_grad_r + m_grad_b

    new_weights = list(current_weights)
    pde_weight = total / (m_grad_r + eps)
    bc_weight  = total / (m_grad_b + eps)

    for i in range(num_pde):
        new_weights[i] = pde_weight
    for i in range(num_pde, num_pde + num_bc):
        new_weights[i] = bc_weight

    return new_weights
