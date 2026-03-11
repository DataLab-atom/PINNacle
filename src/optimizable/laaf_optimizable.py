"""
laaf_optimizable.py  [ReEvo2D 优化目标]

来源：src/model/laaf.py → LAAFlayer.forward
描述：局部自适应激活函数（LAAF）的缩放公式

[背景知识]
LAAF（Jagtap et al., 2020）在标准 Dense 层上引入逐神经元可学习缩放参数 a，
并配合固定超参数 n（放大因子），使每个神经元能自适应调整激活函数的斜率/增益。

[当前实现]
    scaled = n · a ⊙ linear(x)         (⊙ 为逐元素乘)
    out    = activation(scaled)

    其中 a ∈ R^{d_out} 是可学习参数（与层共享），n=10 为固定超参数。

[优化空间]
LLM 可以探索：
  - 不同的缩放形式：softplus(a)、tanh(a)、sigmoid(a) 等（确保缩放有界/正）
  - 归一化变体：layer-norm 后再缩放，防止梯度爆炸
  - n 的自适应策略：将 n 设为可学习参数或随训练步衰减
  - 混合缩放：部分神经元用全局缩放 a_global，部分用局部缩放 a_local

[注意] 请勿修改函数签名（参数名/顺序/类型），否则 LAAFlayer 会出错。
"""

import torch
from torch import Tensor


def laaf_scale(
    linear_out: Tensor,
    a: Tensor,
    n: int = 10,
) -> Tensor:
    """
    对线性层输出施加局部自适应缩放（激活函数输入前的变换）。

    Args:
        linear_out : 线性层输出 W·x + b，shape (..., d_out)
        a          : 逐神经元可学习缩放参数，shape (d_out,)（或可广播形状）
        n          : 固定放大超参数（默认 10，来自原论文设定）

    Returns:
        缩放后的激活输入，shape (..., d_out)，随后会送入 activation()
    """
    return n * torch.mul(a, linear_out)
