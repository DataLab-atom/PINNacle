"""
scripts/_runner.py — 由 run_experiment.py 调用的子进程入口

用法（由 run_experiment._run_scenario 自动调用，勿手动执行）：
    python scripts/_runner.py <root> <patches_json> --pde X --iter N --method M

在 import benchmark_single 之前将 patch 注入 sys.modules，
使 benchmark_single 使用修改后的函数实现，不写任何磁盘文件。
"""

import sys
import ast
import json
import types
import textwrap
from pathlib import Path

root    = Path(sys.argv.pop(1))
patches = json.loads(sys.argv.pop(1))
sys.path.insert(0, str(root))


def _apply_patch(source: str, func_name: str, new_code: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            new_code = textwrap.dedent(new_code).strip() + "\n"
            lines = source.splitlines(keepends=True)
            return "".join(lines[:node.lineno - 1]) + new_code + "\n" + "".join(lines[node.end_lineno:])
    raise ValueError(f"function {func_name!r} not found")


for p in patches:
    fname  = p["file"]
    fpath  = root / "src" / "optimizable" / fname
    source = _apply_patch(fpath.read_text(), p["function"], p["code"])
    mod = types.ModuleType("src.optimizable." + fname.removesuffix(".py"))
    mod.__file__ = str(fpath)
    exec(compile(source, str(fpath), "exec"), mod.__dict__)
    sys.modules[mod.__name__] = mod

import benchmark_single
benchmark_single.main()
