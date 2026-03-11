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

线程安全：注入操作在子进程内存中完成（sys.modules），不写磁盘文件，
ThreadPoolExecutor 并发调用天然无冲突。
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

ROOT        = Path(__file__).parent.parent
OPTIMIZABLE = ROOT / "src" / "optimizable"

# ── 指标正则表达式（匹配 benchmark_single.py 的 stdout）────────────────────────

_FLOAT = r"([\deE+\-\.]+)"
_METRIC_RE = {
    "val_l2re_last": re.compile(rf"Validation L2RE:\s+{_FLOAT}"),
    "final_l2re":    re.compile(rf"Final L2RE:\s+{_FLOAT}"),
}

# ── 子进程注入器：在子进程内将 patch 写入 sys.modules，不碰磁盘 ────────────────
#
# 通过 python -c '_INJECTOR' <root> <patches_json> [benchmark args...] 调用。
# 注入完成后直接调用 benchmark_single.main()，与正常运行完全等价。

_INJECTOR = r"""
import sys, ast, json, types, textwrap
from pathlib import Path

root       = Path(sys.argv.pop(1))
patches    = json.loads(sys.argv.pop(1))

sys.path.insert(0, str(root))

def _patch_source(source, func_name, new_code):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            new_code = textwrap.dedent(new_code).strip() + "\n"
            lines = source.splitlines(keepends=True)
            return "".join(lines[:node.lineno - 1]) + new_code + "\n" + "".join(lines[node.end_lineno:])
    raise ValueError(f"function {func_name!r} not found")

for p in patches:
    fname   = p["file"]
    mname   = "src.optimizable." + fname.removesuffix(".py")
    fpath   = root / "src" / "optimizable" / fname
    source  = _patch_source(fpath.read_text(), p["function"], p["code"])
    mod = types.ModuleType(mname)
    mod.__file__ = str(fpath)
    exec(compile(source, str(fpath), "exec"), mod.__dict__)
    sys.modules[mname] = mod

import benchmark_single
benchmark_single.main()
"""


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
    """将 source 中 func_name 的实现替换为 new_code。"""
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


def _run_scenario(scenario: dict, epochs: Optional[int], replacements: list[dict]) -> dict:
    """
    启动子进程，在其 sys.modules 中注入 patch 后运行 benchmark_single.main()。
    不写任何磁盘文件，多线程并发安全。
    """
    pde    = scenario["pde"]
    method = scenario.get("method", "adam")
    iters  = epochs if epochs is not None else scenario.get("iter", 10000)

    cmd = [
        sys.executable, "-c", _INJECTOR,
        str(ROOT),
        json.dumps(replacements, ensure_ascii=False),
        "--pde",    pde,
        "--iter",   str(iters),
        "--method", method,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    stdout = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr.strip() else "")

    metrics = _parse_metrics(stdout)
    metrics["returncode"] = proc.returncode
    metrics["stdout"]     = stdout

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
    注入新函数实现，运行对应评估场景，返回指标字典。

    Args:
        replacements: 函数替换规格列表，每项：
            {"function": str, "file": str, "code": str}
        scenarios: 评估场景列表，每项：
            {"pde": str, "method": str, "iter": int, "weight": float}
        epochs: 覆盖 scenarios 中的 iter 值。

    Returns:
        {"<pde>_<method>": {"final_l2re": float, "val_l2re_last": float,
                             "returncode": int, "stdout": str}}
    """
    if scenarios is None:
        raise ValueError("scenarios 不能为 None，请由 pinnacle_api.evaluate_funcs 传入")

    for rep in replacements:
        # 提前校验函数名存在（快速失败，避免子进程启动后才报错）
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="注入函数实现并运行 PINNacle 评估场景")
    parser.add_argument("--spec",      required=True, help="替换规格 JSON 文件路径")
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--pretty",    action="store_true")
    parser.add_argument("--no-stdout", dest="no_stdout", action="store_true")
    parser.add_argument("--out",       default=None)
    parser.add_argument("--scenarios", default=None, help="场景 JSON 文件路径")
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
