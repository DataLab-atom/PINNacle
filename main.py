"""
main.py — PINNacle × ReEvo2D 一键启动入口

用 LLM 进化优化 PINNacle 中的核心算法组件，自动搜索更好的：
  - MultiAdam 多损失融合更新策略（multiadam_optimizable.py）
  - NTK 损失权重自适应公式（ntk_optimizable.py）
  - 残差自适应重采样选点策略（sampling_optimizable.py）
  - LRA 边界条件权重更新规则（lra_optimizable.py）

用法
----
# 最简启动（使用环境变量中的 API key）
export OPENAI_API_KEY=your_key_here
python main.py --model deepseek-coder

# 完整参数
python main.py \\
    --model         deepseek-coder \\
    --api-key       $OPENAI_API_KEY \\
    --base-url      https://api.example.com/v1 \\
    --temperature   0.5 \\
    --max-fe        200 \\
    --pop-size      10 \\
    --init-pop-size 20 \\
    --mutation-rate 0.8 \\
    --epochs        5000 \\
    --output-dir    ./runs/pinnacle_reevo

环境变量（可替代命令行参数）
---------------------------
OPENAI_API_KEY   LLM API Key
OPENAI_BASE_URL  API Base URL（如使用第三方代理）
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# ── 路径配置 ─────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.resolve()

# 把 optimizer/ 加入 sys.path，使 reevo2d.py 内的 "from utils.utils import *" 能找到
sys.path.insert(0, str(ROOT_DIR / "optimizer"))
sys.path.insert(0, str(ROOT_DIR))


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PINNacle × ReEvo2D：用 LLM 进化优化 PINN 核心算法组件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # LLM 配置
    p.add_argument("--model",        default="deepseek-coder",
                   help="LLM 模型名，如 gpt-4o / deepseek-coder / GLM-4")
    p.add_argument("--api-key",      default=None,
                   help="LLM API Key（也可通过 OPENAI_API_KEY 环境变量传入）")
    p.add_argument("--base-url",     default=None,
                   help="API Base URL（也可通过 OPENAI_BASE_URL 环境变量传入）")
    p.add_argument("--temperature",  type=float, default=0.5)

    # 进化算法配置
    p.add_argument("--max-fe",       type=int,   default=200,
                   help="最大函数评估次数（停止条件）")
    p.add_argument("--pop-size",     type=int,   default=10,
                   help="每代种群大小")
    p.add_argument("--init-pop-size",type=int,   default=20,
                   help="初始种群大小")
    p.add_argument("--mutation-rate",type=float, default=0.8)

    # PINNacle 评估配置
    p.add_argument("--epochs",       type=int,   default=5000,
                   help="每次评估的训练迭代步数（覆盖各场景默认值）")

    # 输出配置
    p.add_argument("--output-dir",   default="./runs/pinnacle_reevo")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


# ── cfg 构建 ──────────────────────────────────────────────────────────────────

def _build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    problem = SimpleNamespace(
        problem_name="pinnacle",
        description=(
            "Optimize key algorithmic components of Physics-Informed Neural Networks (PINNs) "
            "to minimize the L2 relative error across multiple PDE benchmarks (Burgers, Poisson, Heat). "
            "Target functions span four categories:\n"
            "  1. compute_group_update — MultiAdam multi-loss gradient fusion (multiadam_optimizable.py)\n"
            "  2. adapt_loss_weights   — NTK-based dynamic loss reweighting (ntk_optimizable.py)\n"
            "  3. select_resample_points — Residual-adaptive collocation resampling (sampling_optimizable.py)\n"
            "  4. adapt_boundary_weight  — LRA boundary condition weight update (lra_optimizable.py)\n"
            "All functions are pure Python/PyTorch/NumPy with well-defined signatures and docstrings."
        ),
        problem_size=1,
        obj_type="min",
        problem_type="default",
    )
    return SimpleNamespace(
        algorithm="reevo2d",
        problem=problem,
        reevo_func_index=[0],
        max_fe=args.max_fe,
        pop_size=args.pop_size,
        init_pop_size=args.init_pop_size,
        mutation_rate=args.mutation_rate,
        timeout=900,           # 单次评估超时（秒），3个PDE × 5000步需要时间
        model=args.model,
        temperature=args.temperature,
        diversify_init_pop=True,
    )


# ── LLM 客户端 ────────────────────────────────────────────────────────────────

def _build_client(args: argparse.Namespace):
    model       = args.model
    temperature = args.temperature
    api_key     = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url    = args.base_url or os.environ.get("OPENAI_BASE_URL")

    from utils.llm_client.openai import OpenAIClient

    if model.startswith("GLM"):
        from utils.llm_client.zhipuai import ZhipuAIClient
        return ZhipuAIClient(model, temperature, base_url=base_url, api_key=api_key)
    else:
        # gpt / o1 / o3 / deepseek / gemini / llama 等均走 OpenAI 兼容接口
        return OpenAIClient(model, temperature, base_url=base_url, api_key=api_key)


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(args.output_dir).resolve()
    work_dir   = output_dir / "problems" / "pinnacle"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    logging.info(f"Output directory : {output_dir}")
    logging.info(f"PINNacle root    : {ROOT_DIR}")
    logging.info(f"Model            : {args.model}")
    logging.info(f"Max FE           : {args.max_fe}")
    logging.info(f"Epochs/eval      : {args.epochs}")

    cfg    = _build_cfg(args)
    client = _build_client(args)

    from pinnacle_api import get_funcs, evaluate_funcs
    _evaluate_funcs = functools.partial(evaluate_funcs, epochs=args.epochs)

    from reevo2d import ReEv2d
    optimizer = ReEv2d(
        cfg=cfg,
        root_dir=str(ROOT_DIR),
        generator_llm=client,
        get_funcs=get_funcs,
        evaluate_funcs=_evaluate_funcs,
    )

    best_code, best_code_path = optimizer.evolve()

    logging.info("=" * 60)
    logging.info("Evolution complete.")
    logging.info(f"Best code path : {best_code_path}")
    logging.info("Best code:\n" + str(best_code))


if __name__ == "__main__":
    main()
