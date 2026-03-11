"""
sampling_optimizable.py

残差自适应重采样（RAR）策略的核心选点函数。
决定每轮重采样时从训练集中挑选哪些残差较大的点加入锚点集。

[优化目标]  用更少的额外采样点达到更低的 L2 相对误差（样本效率）。

[外部知识]
- X: 当前训练点坐标，形状 [N, input_dim]
- err: 对应各点的 PDE 残差绝对值，形状 [N]（已取绝对值并降维）
- count: 本轮新增锚点数量（超参数，默认 1）
- 高残差区域 ≠ 高误差区域（残差大的点未必是最难学的点）
- 纯贪心取 top-k 会导致聚集效应（多个点落在同一高残差局部区域）
- 概率采样（按残差分布采样）比确定性 top-k 多样性更好
"""

import numpy as np


def select_resample_points(
    X: np.ndarray,    # 训练点坐标 [N, input_dim]
    err: np.ndarray,  # 各点 PDE 残差绝对值 [N]（已 squeeze）
    count: int,       # 本轮新增锚点数
) -> np.ndarray:
    """
    [RAR-SELECT] 根据 PDE 残差分布，从训练点中选取 count 个点作为新锚点。

    当前实现：确定性 top-k（取残差最大的 count 个点）。
      selected_idx = argsort(err)[-count:]

    可探索方向：
      - 概率采样：以 softmax(err / τ) 为概率分布随机采样（温度 τ 控制集中度）
      - 多样性约束：在高残差中选取彼此距离最大的点（最大最小距离准则）
      - 分层采样：将域划分为子区域，各区域按残差比例分配配额
      - 混合策略：50% top-k + 50% 随机（兼顾聚焦与探索）
      - 课程式采样：早期随机，后期集中高残差（随训练进度调整 τ）

    Args:
        X:     全体训练点坐标
        err:   对应 PDE 残差的绝对值（越大越需要关注）
        count: 新增锚点数量

    Returns:
        np.ndarray: 选中点的坐标，形状 [count, input_dim]
    """
    top_k_idx = np.argsort(err)[-count:]
    return X[top_k_idx]
