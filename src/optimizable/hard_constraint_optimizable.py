"""
hard_constraint_optimizable.py  [ReEvo2D 优化目标]

来源：src/model/hard_constraint.py
描述：初始条件（IC）强制满足的混合衰减公式

[背景知识]
硬约束方法将 PINN 输出构造为：
    u(x, t) = IC(x) · w(t) + NN(x, t) · (1 - w(t))
其中 w(t) 是随时间从 1 衰减到 0 的权重函数。
这样 t=0 时 u = IC（精确满足），t→∞ 时 u = NN（自由度完全还给网络）。

[当前实现]
    w(t) = exp(-α·t)    指数衰减，α=5 为默认衰减速率

[优化空间]
LLM 可以探索：
  - 不同的 w(t) 形式：sigmoid、多项式、tanh、余弦退火
  - 空间自适应衰减：w 同时依赖 x（对异质性介质可能更好）
  - 混合策略：边界和内部用不同的衰减速率
  - 软硬混合：w 下界不趋向 0，留一点 IC 的"影响残余"

[注意] 请勿修改函数签名（参数名/顺序/类型），否则调用方会出错。
"""

import torch
from torch import Tensor


def apply_ic_decay(
    t: Tensor,
    outputs: Tensor,
    ic_values: Tensor,
    alpha: float = 5.0,
) -> Tensor:
    """
    将初始条件与网络输出按时间衰减权重混合。

    Args:
        t         : 时间坐标张量，shape (..., 1)，已归一化到 [0, T]
        outputs   : 网络自由输出，shape (..., n_out)
        ic_values : 初始条件值 IC(x)，shape (..., n_out)
        alpha     : 衰减速率（越大则 IC 约束在更早时刻消退）

    Returns:
        混合后的输出，shape (..., n_out)
        - t=0 时等于 ic_values（精确满足初始条件）
        - t→∞ 时趋近于 outputs（网络自由输出）
    """
    w = torch.exp(-t * alpha)          # IC 权重：t=0→1, t→∞→0
    return outputs * (1 - w) + ic_values * w
