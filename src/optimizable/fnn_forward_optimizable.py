"""
fnn_forward_optimizable.py  [ReEvo2D 优化目标]

来源：src/model/fnn.py → FNN.forward（隐藏层计算部分）
描述：全连接 PINN 网络的核心前向传播逻辑

[背景知识]
标准 FNN 将输入依次通过 L 个 Dense 层 + 激活函数，最后接一个线性输出层：
    h_0 = x
    h_k = activation(W_k · h_{k-1} + b_k),  k = 1, ..., L-1
    y   = W_L · h_{L-1} + b_L

对于 PINN，激活函数选取（tanh / sin / silu）和层间连接方式对
不同频率 PDE 解的收敛速度影响显著（spectral bias 问题）。

[当前实现]
    纯顺序叠加：activation(linear(x)) × (L-1)，再接线性输出层
    ─ 简单可靠，但对高频或多尺度 PDE 可能收敛慢

[优化空间]
LLM 可以探索：
  - 残差连接（ResNet 风格）：每隔 k 层加 skip connection
  - Highway 网络：门控 skip，h = T(x)·H(x) + (1-T(x))·x
  - Modified MLP（Wang et al., 2022）：U/V 分支门控
  - Fourier 特征嵌入：先将输入 x 投影到 sin/cos 频率特征空间
  - 自注意力门控：在某些隐层插入简单 attention 权重重标定

[注意]
  - linears: torch.nn.ModuleList，linears[:-1] 是隐层，linears[-1] 是输出层
  - activation: 一个 callable，接受 Tensor 返回 Tensor
  - 请勿修改函数签名（参数名/顺序/类型），否则 FNN.forward 会出错
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable


def fnn_forward_body(
    x: Tensor,
    linears: "torch.nn.ModuleList",
    activation: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    FNN 的核心前向传播（隐层 + 输出层，不含 input/output transform）。

    Args:
        x          : 网络输入（已经过 input_transform，若有），shape (..., d_in)
        linears    : torch.nn.ModuleList，linears[i] 是第 i 层 Linear 模块
                     linears[:-1] 为隐藏层，linears[-1] 为输出层（无激活）
        activation : 激活函数，callable，接受 Tensor 返回同形状 Tensor

    Returns:
        网络输出（未经过 output_transform），shape (..., d_out)
    """
    for linear in linears[:-1]:
        x = activation(linear(x))
    x = linears[-1](x)
    return x
