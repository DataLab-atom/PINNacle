"""
scripts/run_experiment.py — PINNacle × ReEvo2D 实验执行器

将新函数实现注入 src/optimizable/*_optimizable.py，
运行 benchmark_single.py 对应场景，解析指标，完成后自动还原所有文件。

与 PI-GANO 的 run_experiment.py 接口保持一致，供 pinnacle_api.evaluate_funcs() 调用。

━━━━━━━━━ 作为模块调用 ━━━━━━━━━
    from scripts.run_experiment import run_experiment

    results = run_experiment(
        replacements=[{
            "function": "compute_group_update",
            "file":     "multiadam_optimizable.py",
            "code":     \"""
def compute_group_update(exp_avg, denom, group_weights):
    # 改进版：归一化后再加权
    update_raw = exp_avg / denom
    norms = update_raw.norm(dim=list(range(1, update_raw.dim())), keepdim=True).clamp(min=1e-8)
    w = group_weights.view((-1,) + (1,) * (update_raw.dim()-1))
    return (update_raw / norms * w).sum(dim=0)
\""",
        }],
        scenarios=[{"pde": "Burgers1D", "iter": 5000, "method": "multiadam", "weight": 1.0}],
        epochs=5000,
    )
    # results["Burgers1D_multiadam"]["final_l2re"]  → float

━━━━━━━━━ 命令行调用 ━━━━━━━━━
    python scripts/run_experiment.py --spec my_exp.json --pretty

替换规格 JSON 格式：
[{"function": "compute_group_update", "file": "multiadam_optimizable.py", "code": "..."}]

输出格式（每个场景）：
{
  "<pde>_<method>": {
    "final_l2re":       float,   # 最佳 L2 相对误差（主要指标）
    "val_l2re_last":    float,   # 最后一次验证 L2RE
    "returncode":       int,     # 0 = 成功
    "stdout":           str,
  }
}
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# ── 路径常量 ──────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
OPTIMIZABLE  = ROOT / "src" / "optimizable"

# ── 指标正则表达式（匹配 benchmark_single.py 的 stdout）────────────────────────

_FLOAT = r"([\deE+\-\.]+)"
_METRIC_RE = {
    "val_l2re_last": re.compile(rf"Validation L2RE:\s+{_FLOAT}"),
    "final_l2re":    re.compile(rf"Final L2RE:\s+{_FLOAT}"),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  核心工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_function_range(source: str, func_name: str) -> tuple[int, int]:
    """返回函数在源码中的行范围（1-indexed，含首尾）。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.lineno, node.end_lineno
    raise ValueError(f"函数 '{func_name}' 未在源码中找到")


def _patch_source(source: str, func_name: str, new_code: str) -> str:
    """将 source 中 func_name 的实现替换为 new_code（保留其前的模块级代码）。"""
    start, end = _find_function_range(source, func_name)
    new_code = textwrap.dedent(new_code).strip() + "\n"
    lines = source.splitlines(keepends=True)
    return "".join(lines[: start - 1]) + new_code + "\n" + "".join(lines[end:])


def _parse_metrics(stdout: str) -> dict:
    """从 stdout 中提取所有已知指标，取每类最后一次出现的值。"""
    metrics: dict[str, float] = {}
    for name, pattern in _METRIC_RE.items():
        matches = pattern.findall(stdout)
        if matches:
            try:
                metrics[name] = float(matches[-1])
            except ValueError:
                pass
    return metrics


def _run_scenario(scenario: dict, epochs: Optional[int]) -> dict:
    """运行单个场景（benchmark_single.py 子进程），返回指标 + returncode + stdout。"""
    pde    = scenario["pde"]
    method = scenario.get("method", "adam")
    iters  = epochs if epochs is not None else scenario.get("iter", 10000)

    cmd = [
        sys.executable, str(ROOT / "benchmark_single.py"),
        "--pde",    pde,
        "--iter",   str(iters),
        "--method", method,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    stdout = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr.strip() else "")

    metrics = _parse_metrics(stdout)
    metrics["returncode"] = proc.returncode
    metrics["stdout"]     = stdout

    # 如果没有 final_l2re（训练崩溃），设为 inf
    if "final_l2re" not in metrics:
        metrics["final_l2re"] = float("inf")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  公开 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(
    replacements: list[dict],
    scenarios: Optional[list[dict]] = None,
    epochs: Optional[int] = None,
) -> dict:
    """
    注入新函数实现，运行对应评估场景，返回指标字典，并还原所有文件。

    Args:
        replacements: 函数替换规格列表，每项：
            {
              "function": str,   # 函数名（须与 *_optimizable.py 中精确匹配）
              "file":     str,   # 文件名（仅文件名，不含路径）
              "code":     str,   # 完整函数定义字符串
            }
        scenarios: 评估场景列表，每项：
            {
              "pde":    str,    # PDE 类名（benchmark_single.py --pde 参数）
              "method": str,    # 训练方法（adam / multiadam / ntk / lra / rar）
              "iter":   int,    # 迭代步数（被 epochs 参数覆盖）
              "weight": float,  # 场景权重（供 evaluate_funcs 聚合）
            }
        epochs: 覆盖 scenarios 中的 iter 值；None = 使用各场景默认值。

    Returns:
        dict，键为 "<pde>_<method>"，值为：
        {
            "final_l2re":    float,  # 最佳 L2 相对误差（主要指标）
            "val_l2re_last": float,  # 最后一次验证 L2RE
            "returncode":    int,    # 0 = 成功
            "stdout":        str,
        }

    Raises:
        ValueError: 函数名不存在于目标文件。
        RuntimeError: 文件注入失败（已自动还原）。
    """
    if scenarios is None:
        raise ValueError("scenarios 不能为 None，请由 pinnacle_api.evaluate_funcs 传入")

    # 1. 注入函数（原子操作，失败即还原）
    backups: dict[Path, str] = {}
    try:
        for rep in replacements:
            fpath   = OPTIMIZABLE / rep["file"]
            current = fpath.read_text(encoding="utf-8")
            if fpath not in backups:
                backups[fpath] = current
            patched = _patch_source(current, rep["function"], rep["code"])
            fpath.write_text(patched, encoding="utf-8")
            print(f"[patch] {rep['file']}::{rep['function']}", file=sys.stderr)
    except Exception as exc:
        for path, orig in backups.items():
            path.write_text(orig, encoding="utf-8")
        raise RuntimeError(f"注入失败，已还原所有文件。原因：{exc}") from exc

    # 2. 运行各场景
    results: dict[str, dict] = {}
    try:
        for sc in scenarios:
            key = f"{sc['pde']}_{sc.get('method', 'adam')}"
            print(f"[run]   {key}, epochs={epochs or sc.get('iter', '?')}", file=sys.stderr)
            results[key] = _run_scenario(sc, epochs)
            rc    = results[key]["returncode"]
            l2re  = results[key].get("final_l2re", "N/A")
            print(f"[done]  {key}, returncode={rc}, final_l2re={l2re}", file=sys.stderr)
    finally:
        # 3. 无论是否出错，还原所有文件
        for path, orig in backups.items():
            path.write_text(orig, encoding="utf-8")
        print("[restore] 所有 optimizable 文件已还原", file=sys.stderr)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="注入函数实现并运行 PINNacle 评估场景")
    parser.add_argument("--spec",    required=True, help="替换规格 JSON 文件路径")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--pretty",  action="store_true")
    parser.add_argument("--no-stdout", dest="no_stdout", action="store_true")
    parser.add_argument("--out",     default=None)
    # 场景可通过 JSON 文件中的 "scenarios" 字段指定，或使用默认 3 个场景
    parser.add_argument("--scenarios", default=None,
                        help="场景 JSON 文件路径（若不指定则使用默认 3 个评估场景）")
    args = parser.parse_args()

    from pinnacle_api import _EVAL_SCENARIOS
    replacements = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    scenarios    = json.loads(Path(args.scenarios).read_text()) if args.scenarios else _EVAL_SCENARIOS

    results = run_experiment(replacements, scenarios=scenarios, epochs=args.epochs)

    if args.no_stdout:
        for v in results.values():
            v.pop("stdout", None)

    output = json.dumps(results, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"结果已写入 {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
