"""
benchmark_single.py — 单 PDE 快速评估脚本

供 scripts/run_experiment.py 以子进程形式调用。
仅运行指定的 PDE 案例，输出标准化指标行，供父进程正则解析。

用法（通常由 run_experiment.py 自动调用，也可手动运行）：
    python benchmark_single.py --pde Burgers1D --iter 10000 --method adam
    python benchmark_single.py --pde Poisson2D_Classic --iter 5000 --method multiadam
    python benchmark_single.py --pde Heat2D_Multiscale --iter 8000

支持的 --pde 值（与 benchmark.py 中 pde_list 对应的类名）：
    Burgers1D, Burgers2D,
    Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea,
    Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime,
    NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime,
    Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime,
    KuramotoSivashinskyEquation, GrayScottEquation,
    PoissonND, HeatND

输出格式（父进程 _METRIC_RE 正则匹配目标）：
    Validation L2RE: <float>     ← 每 log_every 轮打印一次
    Final L2RE: <float>          ← 训练结束时打印最佳值
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime, HeatND
from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.rar import rar_wrapper

# ── PDE 名称 → 类的映射 ──────────────────────────────────────────────────────

PDE_MAP = {
    "Burgers1D":                   Burgers1D,
    "Burgers2D":                   Burgers2D,
    "Poisson2D_Classic":           Poisson2D_Classic,
    "PoissonBoltzmann2D":          PoissonBoltzmann2D,
    "Poisson3D_ComplexGeometry":   Poisson3D_ComplexGeometry,
    "Poisson2D_ManyArea":          Poisson2D_ManyArea,
    "Heat2D_VaryingCoef":          Heat2D_VaryingCoef,
    "Heat2D_Multiscale":           Heat2D_Multiscale,
    "Heat2D_ComplexGeometry":      Heat2D_ComplexGeometry,
    "Heat2D_LongTime":             Heat2D_LongTime,
    "NS2D_LidDriven":              NS2D_LidDriven,
    "NS2D_BackStep":               NS2D_BackStep,
    "NS2D_LongTime":               NS2D_LongTime,
    "Wave1D":                      Wave1D,
    "Wave2D_Heterogeneous":        Wave2D_Heterogeneous,
    "Wave2D_LongTime":             Wave2D_LongTime,
    "KuramotoSivashinskyEquation": KuramotoSivashinskyEquation,
    "GrayScottEquation":           GrayScottEquation,
    "PoissonND":                   PoissonND,
    "HeatND":                      HeatND,
}


# ── 评估回调（内联，不依赖 trainer.py 的文件 I/O） ────────────────────────────

class InlineMetricCallback(dde.callbacks.Callback):
    """每 log_every 轮打印 L2RE，训练结束时打印 Final L2RE。"""

    def __init__(self, log_every: int = 500):
        super().__init__()
        self.log_every = log_every
        self.epoch = 0
        self.best_l2re = float("inf")

    def _compute_l2re(self) -> float | None:
        pde = self.model.pde
        if pde.ref_sol is not None:
            sample_points = 2500 if pde.input_dim == 2 else 20000
            geom = self.model.data.geom
            sample_fn = getattr(geom, "uniform_points", geom.random_points)
            test_x = sample_fn(sample_points, boundary=False)
            test_y = pde.ref_sol(test_x)
        elif pde.ref_data is not None:
            nan_mask = np.isnan(pde.ref_data).any(axis=1)
            test_x = pde.ref_data[~nan_mask, :pde.input_dim]
            test_y = pde.ref_data[~nan_mask, pde.input_dim:]
        else:
            return None

        with torch.no_grad():
            pred = self.model.predict(test_x)
        l2re = float(np.sqrt(((pred - test_y) ** 2).mean()) / (np.sqrt((test_y ** 2).mean()) + 1e-8))
        return l2re

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch % self.log_every != 0:
            return
        l2re = self._compute_l2re()
        if l2re is None:
            return
        self.best_l2re = min(self.best_l2re, l2re)
        print(f"Validation L2RE: {l2re:.6f}", flush=True)

    def on_train_end(self):
        l2re = self._compute_l2re()
        if l2re is not None:
            self.best_l2re = min(self.best_l2re, l2re)
        print(f"Final L2RE: {self.best_l2re:.6f}", flush=True)


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PINNacle single-PDE benchmark")
    parser.add_argument("--pde",          type=str,   required=True,  help="PDE 类名（见 PDE_MAP）")
    parser.add_argument("--iter",         type=int,   default=10000,  help="训练迭代步数")
    parser.add_argument("--method",       type=str,   default="adam", help="优化方法")
    parser.add_argument("--hidden-layers",type=str,   default="100*5")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--log-every",    type=int,   default=500)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--loss-weight",  type=str,   default="")
    parser.add_argument("--device",       type=str,   default="cpu")
    args = parser.parse_args()

    # 设备
    if args.device != "cpu":
        torch.cuda.set_device(int(args.device))
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(args.seed)

    if args.pde not in PDE_MAP:
        print(f"[ERROR] 未知 PDE: {args.pde}. 可选: {list(PDE_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    pde = PDE_MAP[args.pde]()

    # 网络
    hidden = parse_hidden_layers(args)
    if args.method == "laaf":
        net = DNN_LAAF(len(hidden) - 1, hidden[0], pde.input_dim, pde.output_dim)
    elif args.method == "gaaf":
        net = DNN_GAAF(len(hidden) - 1, hidden[0], pde.input_dim, pde.output_dim)
    else:
        net = dde.nn.FNN([pde.input_dim] + hidden + [pde.output_dim], "tanh", "Glorot normal")
    net = net.float()

    # 损失权重
    loss_weights = parse_loss_weight(args)
    if loss_weights is None:
        loss_weights = np.ones(pde.num_loss)
    else:
        loss_weights = np.array(loss_weights)

    # 优化器
    if args.method == "multiadam":
        opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99),
                        loss_group_idx=[pde.num_pde])
    elif args.method == "lra":
        base_opt = torch.optim.Adam(net.parameters(), args.lr)
        opt = LR_Adaptor(base_opt, loss_weights, pde.num_pde)
    elif args.method == "ntk":
        base_opt = torch.optim.Adam(net.parameters(), args.lr)
        opt = LR_Adaptor_NTK(base_opt, loss_weights, pde)
    elif args.method == "lbfgs":
        opt = Adam_LBFGS(net.parameters(), switch_epoch=5000, adam_param={"lr": args.lr})
    else:
        opt = torch.optim.Adam(net.parameters(), args.lr)

    if args.method == "gepinn":
        pde.use_gepinn()

    model = pde.create_model(net)
    model.compile(opt, loss_weights=loss_weights)

    metric_cb = InlineMetricCallback(log_every=args.log_every)
    train_kwargs = dict(
        iterations=args.iter,
        display_every=args.log_every,
        callbacks=[metric_cb],
    )

    if args.method == "rar":
        model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})

    save_path = f"runs/reevo_eval/{args.pde}_{int(time.time())}"
    os.makedirs(save_path, exist_ok=True)
    model.train(**train_kwargs, model_save_path=save_path)


if __name__ == "__main__":
    main()
