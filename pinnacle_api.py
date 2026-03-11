"""
pinnacle_api.py — ReEvo2D ↔ PINNacle 评估接口

提供两个核心 API（与 PI-GANO 的 u2e_api.py 接口完全兼容）：

    get_funcs()          → 读取 src/optimizable/ 中所有可优化函数的元数据
    evaluate_funcs(...)  → 注入新函数实现、运行评估场景、返回指标评分

运行时由 main.py 通过 functools.partial 传入 ReEvo2D。

评估场景（默认）：Burgers1D + Poisson2D_Classic + Heat2D_Multiscale
这三个场景覆盖：时间依赖 PDE、稳态椭圆、多尺度问题，
且训练迭代数较少（5000步），保证进化闭环在合理时间内完成。
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Optional

ROOT         = Path(__file__).parent
OPTIMIZABLE  = ROOT / "src" / "optimizable"

# ── 评估场景：每场景在 benchmark_single.py 中对应一个 --pde 参数 ──────────────

_EVAL_SCENARIOS: list[dict] = [
    {"pde": "Burgers1D",         "iter": 5000,  "method": "adam",      "weight": 1.0},
    {"pde": "Poisson2D_Classic", "iter": 5000,  "method": "adam",      "weight": 1.0},
    {"pde": "Heat2D_Multiscale", "iter": 5000,  "method": "adam",      "weight": 1.0},
]

# 函数所属文件 → 受影响场景（与 PI-GANO run_experiment.py 同构）
_FUNC_FILE_TO_SCENARIOS: dict[str, list[dict]] = {
    "multiadam_optimizable.py": [{"pde": "Burgers1D",         "iter": 5000, "method": "multiadam", "weight": 1.0},
                                  {"pde": "Poisson2D_Classic", "iter": 5000, "method": "multiadam", "weight": 1.0}],
    "ntk_optimizable.py":       [{"pde": "Burgers1D",         "iter": 5000, "method": "ntk",       "weight": 1.0},
                                  {"pde": "Poisson2D_Classic", "iter": 5000, "method": "ntk",       "weight": 1.0}],
    "sampling_optimizable.py":  [{"pde": "Burgers1D",         "iter": 5000, "method": "rar",       "weight": 1.0},
                                  {"pde": "Heat2D_Multiscale", "iter": 5000, "method": "rar",       "weight": 1.0}],
    "lra_optimizable.py":       [{"pde": "Burgers1D",         "iter": 5000, "method": "lra",       "weight": 1.0},
                                  {"pde": "Poisson2D_Classic", "iter": 5000, "method": "lra",       "weight": 1.0}],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  get_funcs()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_optimizable_file(path: Path) -> list[dict]:
    """解析单个 *_optimizable.py，提取公开函数的元数据。"""
    source = path.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    items  = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):   # 跳过私有函数
            continue

        # 重建函数源码（从原始文件提取对应行）
        lines = source.splitlines()
        func_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])

        # 签名
        args_info = []
        for arg in node.args.args:
            annotation = ast.unparse(arg.annotation) if arg.annotation else None
            args_info.append({"name": arg.arg, "type": annotation})
        return_type = ast.unparse(node.returns) if node.returns else None

        # docstring
        docstring  = ast.get_docstring(node) or ""
        first_line = docstring.strip().split("\n")[0] if docstring else node.name

        items.append({
            "func_name":        node.name,
            "func_source":      func_src,
            "func_description": first_line,
            "doc":              _build_markdown_doc(path.name, node.name, args_info, return_type, docstring),
            "file":             path.name,
            "args":             args_info,
            "returns":          return_type,
            "is_modified":      False,
        })

    return items


def _build_markdown_doc(filename: str, func_name: str, args: list, returns: Optional[str], docstring: str) -> str:
    """将函数元数据格式化为 Markdown 文档（供 LLM user_generator prompt 使用）。"""
    sig_parts = ", ".join(
        f"{a['name']}: {a['type']}" if a["type"] else a["name"]
        for a in args
    )
    ret = f" -> {returns}" if returns else ""
    lines = [
        f"## `{func_name}` in `{filename}`",
        "",
        f"```python",
        f"def {func_name}({sig_parts}){ret}:",
        f'    """',
        *[f"    {l}" for l in docstring.splitlines()],
        f'    """',
        f"```",
        "",
        "**Full docstring:**",
        "",
        docstring,
    ]
    return "\n".join(lines)


def get_funcs() -> list[dict]:
    """
    读取 src/optimizable/ 下所有 *_optimizable.py 文件，
    返回 ReEvo2D 需要的函数元数据列表。

    返回格式（每项）：
    {
        "func_name":        str,   # 函数名
        "func_source":      str,   # 完整函数源码
        "func_description": str,   # 单行描述（docstring 首行）
        "doc":              str,   # Markdown 格式的完整文档
        "file":             str,   # 所在文件名（仅文件名，不含路径）
        "args":             list,  # [{"name": ..., "type": ...}, ...]
        "returns":          str,   # 返回类型注解
        "is_modified":      bool,  # ReEvo2D 用于标记是否被替换
    }
    """
    files = sorted(OPTIMIZABLE.glob("*_optimizable.py"))
    if not files:
        raise RuntimeError(f"未找到任何 *_optimizable.py 文件（路径：{OPTIMIZABLE}）")

    result = []
    for f in files:
        result.extend(_parse_optimizable_file(f))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  evaluate_funcs()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_funcs(
    func_list: list[dict],
    epochs: Optional[int] = None,
) -> list[dict]:
    """
    注入新函数实现，运行评估场景，返回评分列表（供 ReEvo2D._aggregate_scores 使用）。

    Args:
        func_list: get_funcs() 返回的函数列表，其中被修改的项 is_modified=True。
                   ReEvo2D 会在调用前将新实现写入 func_source。
        epochs:    覆盖默认迭代步数（None = 使用 _EVAL_SCENARIOS 默认值）

    Returns:
        list[dict]，每项格式：
        {
            "name":      str,    # 指标名，如 "L2RE_Burgers1D_multiadam"
            "value":     float,  # L2 相对误差（越小越好）
            "direction": "min",
            "weight":    float,  # 场景权重
        }

    Raises:
        RuntimeError: 所有场景均失败时抛出。
    """
    from scripts.run_experiment import run_experiment

    # 找出被修改的函数及其所属文件
    modified = [f for f in func_list if f.get("is_modified", False)]

    # 推断受影响的评估场景
    active_scenarios: list[dict] = []
    seen_keys: set[tuple] = set()
    for func_dict in modified:
        fname = func_dict["file"]
        scenarios = _FUNC_FILE_TO_SCENARIOS.get(fname, _EVAL_SCENARIOS)
        for sc in scenarios:
            key = (sc["pde"], sc["method"])
            if key not in seen_keys:
                active_scenarios.append(sc)
                seen_keys.add(key)

    if not active_scenarios:
        active_scenarios = _EVAL_SCENARIOS

    # 构建替换规格
    replacements = [
        {
            "function": f["func_name"],
            "file":     f["file"],
            "code":     f["func_source"],
        }
        for f in modified
    ]

    # 运行实验（注入 → 训练 → 还原）
    results = run_experiment(
        replacements=replacements,
        scenarios=active_scenarios,
        epochs=epochs,
    )

    # 转换为 ReEvo2D 评分格式
    scores: list[dict] = []
    for sc in active_scenarios:
        key      = f"{sc['pde']}_{sc['method']}"
        l2re     = results.get(key, {}).get("final_l2re", float("inf"))
        scores.append({
            "name":      f"L2RE_{key}",
            "value":     l2re,
            "direction": "min",
            "weight":    sc["weight"],
        })

    if all(s["value"] == float("inf") for s in scores):
        raise RuntimeError("所有评估场景均失败，请检查注入的函数实现")

    return scores
