"""
input_encoding_optimizable.py  [ReEvo2D 优化目标]

来源：src/model/fnn.py → FNN.forward（输入映射步骤）
描述：网络输入坐标的特征编码（谱偏置缓解）

[背景知识]
标准 MLP 对高频函数存在"谱偏置"（spectral bias / frequency principle）：
网络倾向于先学习低频分量，高频解收敛极慢甚至无法收敛。

Fourier 特征映射（Tancik et al., 2020; Wang et al., 2021）通过将输入坐标
映射到随机 / 可学习的频率空间，显著缓解谱偏置：
    γ(x) = [cos(2π·B·x), sin(2π·B·x)]
其中 B 是固定随机矩阵（Random Fourier Features）或可学习矩阵。

对 PINN 而言，Modified MLP（Wang et al., 2022）也属于输入编码的一种变体：
    U = sigmoid(W₁·x + b₁)
    V = sigmoid(W₂·x + b₂)
然后在每个隐层计算 H = (1 - gate) * H_prev + gate * UV

[当前实现]
    encode_input 默认返回恒等变换（不修改输入），与原始 FNN 行为完全一致

[优化空间]
LLM 可以探索：
  - Random Fourier Features：固定高斯随机矩阵 B，σ 为尺度超参数
  - Learnable Fourier Features：可学习的频率矩阵（需在 FNN 中注册为参数）
  - Positional Encoding（NERF 风格）：sin/cos 多尺度位置编码
  - 输入归一化：按 bbox 将坐标归一化到 [-1, 1]
  - 多尺度输入拼接：同时保留原始坐标和编码坐标

[注意]
  - 输出维度可以与输入维度不同，但 FNN 的第一个 Linear 层需适配
  - 当前接口设计：LLM 可以返回不同维度，FNN 的线性层会自动匹配
    （前提：LLM 生成的代码同时调整第一层权重，或者只返回同维特征）
  - 最安全的探索方向：保持输出维度 = 输入维度（直接替换坐标）
  - 进阶探索：返回 2 × input_dim 维（sin + cos 拼接），需配合 FNN 层尺寸调整

[注意] 请勿修改函数签名（参数名/类型），否则 FNN.forward 会出错。
"""

import math
import torch
from torch import Tensor


def encode_input(x: Tensor, input_dim: int) -> Tensor:
    """
    对输入坐标施加特征编码，在进入网络隐层前变换表示空间。

    Args:
        x         : 输入坐标张量，shape (..., input_dim)
                    对时变问题通常为 (x₁, x₂, ..., t)，最后一列是时间
        input_dim : 输入坐标维度（= x.shape[-1]）

    Returns:
        编码后的特征，shape (..., output_dim)
        - 默认：output_dim = input_dim（恒等变换，兼容原 FNN）
        - 若返回其他维度：需确保 FNN 第一层 Linear 的 in_features 匹配
    """
    return x  # 默认：恒等变换，与原始 FNN 行为完全一致
