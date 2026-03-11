"""
scripts/run_experiment.py — PINNacle × ReEvo2D 实验执行器

将新函数实现注入 src/optimizable/*_optimizable.py，
运行 benchmark_single.py 对应场景，解析指标。

与 PI-GANO 的 run_experiment.py 接口保持一致，供 pinnacle_api.evaluate_funcs() 调用。

━━━━━━━━━ 作为模块调用 ━━━━━━━━━
    from scripts.run_experiment import run_experiment

    results = run_experiment(
        replacements=[{
            "function": "compute_group_update",
            "file":     "multiadam_optimizable.py",
            "code":     \"""
def compute_group_update(exp_avg, denom, group_weights):
    update_raw = exp_avg / denom
    norms = update_raw.norm(dim=list(range(1, update_raw.dim())), keepdim=True).clamp(min=1e-8)
    w = group_weights.view((-1,) + (1,) * (update_raw.dim()-1))
    return (update_raw / norms * w).sum(dim=0)
\""",
        }],
        scenarios=[{"pde": "Burgers1D", "iter": 5000, "method": "multiadam", "weight": 1.0}],
        epochs=5000,
    )

━━━━━━━━━ 命令行调用 ━━━━━━━━━
    python scripts/run_experiment.py --spec my_exp.json --pretty
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

ROOT        = Path(__file__).parent.parent
OPTIMIZABLE = ROOT / "src" / "optimizable"
RUNNER      = Path(__file__).parent / "_runner.py"

_FLOAT = r"([\deE+\-\.]+)"
_METRIC_RE = {
    "val_l2re_last": re.compile(rf"Validation L2RE:\s+{_FLOAT}"),
    "final_l2re":    re.compile(rf"Final L2RE:\s+{_FLOAT}"),
}


def _find_function_range(source: str, func_name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.lineno, node.end_lineno
    raise ValueError(f"函数 '{func_name}' 未在源码中找到")


def _parse_metrics(stdout: str) -> dict:
    metrics: dict[str, float] = {}
    for name, pattern in _METRIC_RE.items():
        matches = pattern.findall(stdout)
        if matches:
            try:
                metrics[name] = float(matches[-1])
            except ValueError:
                pass
    return metrics


def _run_scenario(scenario: dict, epochs: Optional[int], replacements: list[dict]) -> dict:
    pde    = scenario["pde"]
    method = scenario.get("method", "adam")
    iters  = epochs if epochs is not None else scenario.get("iter", 10000)

    cmd = [
        sys.executable, str(RUNNER),
        str(ROOT), json.dumps(replacements, ensure_ascii=False),
        "--pde", pde, "--iter", str(iters), "--method", method,
    ]
    proc   = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    stdout = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr.strip() else "")

    metrics = _parse_metrics(stdout)
    metrics["returncode"] = proc.returncode
    metrics["stdout"]     = stdout
    metrics.setdefault("final_l2re", float("inf"))
    return metrics


def run_experiment(
    replacements: list[dict],
    scenarios: Optional[list[dict]] = None,
    epochs: Optional[int] = None,
) -> dict:
    """
    注入新函数实现，运行对应评估场景，返回指标字典。

    Args:
        replacements: [{"function": str, "file": str, "code": str}, ...]
        scenarios:    [{"pde": str, "method": str, "iter": int, "weight": float}, ...]
        epochs:       覆盖 scenarios 中的 iter 值。

    Returns:
        {"<pde>_<method>": {"final_l2re": float, "val_l2re_last": float,
                             "returncode": int, "stdout": str}}
    """
    if scenarios is None:
        raise ValueError("scenarios 不能为 None，请由 pinnacle_api.evaluate_funcs 传入")

    # 提前校验函数名，快速失败
    for rep in replacements:
        source = (OPTIMIZABLE / rep["file"]).read_text(encoding="utf-8")
        _find_function_range(source, rep["function"])
        print(f"[patch] {rep['file']}::{rep['function']}", file=sys.stderr)

    results: dict[str, dict] = {}
    for sc in scenarios:
        key = f"{sc['pde']}_{sc.get('method', 'adam')}"
        print(f"[run]   {key}, epochs={epochs or sc.get('iter', '?')}", file=sys.stderr)
        results[key] = _run_scenario(sc, epochs, replacements)
        rc   = results[key]["returncode"]
        l2re = results[key].get("final_l2re", "N/A")
        print(f"[done]  {key}, returncode={rc}, final_l2re={l2re}", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(description="注入函数实现并运行 PINNacle 评估场景")
    parser.add_argument("--spec",      required=True)
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--pretty",    action="store_true")
    parser.add_argument("--no-stdout", dest="no_stdout", action="store_true")
    parser.add_argument("--out",       default=None)
    parser.add_argument("--scenarios", default=None)
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
