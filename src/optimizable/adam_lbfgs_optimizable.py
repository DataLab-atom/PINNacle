"""
adam_lbfgs_optimizable.py  [ReEvo2D 优化目标]

来源：src/optimizer/adam_lbfgs.py → Adam_LBFGS.step
描述：Adam → L-BFGS 切换决策策略

[背景知识]
Adam+L-BFGS 两阶段训练是 PINN 中常用的组合策略：
  - Adam：前期快速下降，跳出局部极值，对噪声梯度鲁棒
  - L-BFGS：后期精细优化，利用曲率信息快速收敛到高精度解

[当前实现]
    在固定迭代步数 switch_epoch 处做一次性硬切换：
      step < switch_epoch  → 使用 Adam
      step >= switch_epoch → 使用 L-BFGS

[优化空间]
LLM 可以探索：
  - 基于损失平台检测的自适应切换（连续 K 步相对改变量 < ε 时切换）
  - 基于梯度范数的切换（梯度模 < 阈值时切换，说明 Adam 已达平台）
  - 余弦退火式软切换（以概率从 Adam 过渡到 L-BFGS）
  - 重复切换策略（L-BFGS 卡住后退回 Adam 热身再切换）
  - 自适应 switch_epoch：根据问题维度/网络深度动态计算切换时机

[调用约定]
  - 返回 'adam' 表示本步使用 Adam，返回 'lbfgs' 表示使用 L-BFGS
  - loss_history: 最近若干步的 loss 值列表（最新在末尾），初始为空列表
  - grad_norm: 当前参数梯度的 L2 范数（由调用方在调用前计算）

[注意] 请勿修改函数签名，否则 Adam_LBFGS.step 会出错。
"""

from __future__ import annotations


def select_optimizer(
    current_step: int,
    switch_epoch: int,
    loss_history: list[float],
    grad_norm: float,
) -> str:
    """
    决定当前迭代步使用哪个优化器。

    Args:
        current_step  : 当前训练步（从 1 开始）
        switch_epoch  : 用户配置的切换步数阈值
        loss_history  : 最近步的 loss 值列表，最新值在末尾（可能为空列表）
        grad_norm     : 当前所有参数梯度的 L2 范数

    Returns:
        'adam'  — 本步使用 Adam 优化器
        'lbfgs' — 本步使用 L-BFGS 优化器
    """
    if current_step < switch_epoch:
        return 'adam'
    return 'lbfgs'
