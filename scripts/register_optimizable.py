"""
scripts/register_optimizable.py — 可优化函数注册工具

让用户自主决定"哪些函数交给 LLM 进化优化"，并自动完成：
  1. 从源文件提取目标函数
  2. 生成 src/optimizable/*_optimizable.py 包装文件
  3. 在源文件中注入 import + 替换调用（若函数尚未接管）

子命令
------
  --list          列出 optimizable_config.yaml 中的所有候选函数
  --status        显示当前已注册函数的状态（文件存在? 调用已接管?）
  --apply         根据 registered 区块执行全量注册（幂等）
  --add           将新函数添加到 registered 并立即 apply
  --remove        从 registered 中注销函数（不删除生成文件，需手动还原 import）

示例
----
  # 查看所有候选
  python scripts/register_optimizable.py --list

  # 检查注册状态
  python scripts/register_optimizable.py --status

  # 应用当前配置（首次 / 修改 yaml 后运行）
  python scripts/register_optimizable.py --apply

  # 新增候选并立即激活
  python scripts/register_optimizable.py --add \\
      --source src/optimizer/multiadam.py \\
      --function sadam \\
      --output multiadam_sadam_optimizable \\
      --desc "完整 MultiAdam 参数更新循环" \\
      --eval-pde Burgers1D --eval-method multiadam --eval-iter 5000

  # 注销（不会删除已生成文件）
  python scripts/register_optimizable.py --remove --function sadam
"""

from __future__ import annotations

import ast
import argparse
import sys
import textwrap
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[ERROR] 缺少 pyyaml，请先运行: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "optimizable_config.yaml"
OPTIMIZABLE = ROOT / "src" / "optimizable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  配置文件读写
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("registered", [])
    cfg.setdefault("candidates", [])
    return cfg


def _save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False, width=120)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AST 工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_function(source_path: Path, func_name: str) -> str:
    """从源文件中提取指定函数的完整源码（含 docstring）。"""
    source = source_path.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    lines  = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    raise ValueError(
        f"函数 '{func_name}' 未在 {source_path} 中找到。\n"
        f"可用函数：{_list_public_functions(source_path)}"
    )


def _list_public_functions(source_path: Path) -> list[str]:
    """列出文件中所有公开函数名。"""
    source = source_path.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    return [
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    ]


def _build_optimizable_module(
    func_name: str,
    func_source: str,
    description: str,
    source_file: str,
) -> str:
    """生成标准格式的 *_optimizable.py 文件内容。"""
    header = textwrap.dedent(f"""\
        \"\"\"
        {func_name}_optimizable.py  [由 register_optimizable.py 自动生成]

        来源：{source_file}
        描述：{description}

        [优化指引]
        此文件中的函数将被 ReEvo2D 进化优化器反复替换和评估。
        你可以：
          1. 手动修改此文件中的函数实现，作为新的 seed（初始方案）
          2. 在 optimizable_config.yaml 的 registered 条目中调整 eval_scenarios
          3. 运行 python main.py 让 LLM 自动搜索更优实现

        [注意] 请勿修改函数签名（参数名/类型/返回类型），否则调用方会出错。
        \"\"\"

        import math
        import numpy as np
        import torch
        from torch import Tensor


    """)
    return header + func_source + "\n"


def _is_import_present(source: str, module_path: str, func_name: str) -> bool:
    """检查源文件是否已包含从 optimizable 模块的 import。"""
    return f"from {module_path} import {func_name}" in source


def _inject_import(source: str, module_path: str, func_name: str) -> str:
    """在文件顶部 import 区块末尾注入 optimizable import。"""
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    import_line = f"from {module_path} import {func_name}  # [OPTIMIZABLE]\n"
    lines.insert(insert_at, import_line)
    return "".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  核心操作：注册单个函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def register_one(entry: dict, verbose: bool = True) -> None:
    """
    根据 registered 条目执行注册：
      1. 若 output_file 不存在：从 source_file 提取函数 → 生成 optimizable 文件
      2. 若 source_file 未 import：注入 import 语句（仅提示，不自动修改调用点）
    """
    source_file = ROOT / entry["source_file"]
    output_file = ROOT / entry["output_file"]
    func_name   = entry["function"]
    description = entry.get("description", func_name)

    # ── Step 1: 若 optimizable 文件不存在，则生成 ─────────────────────────────
    if not output_file.exists():
        if verbose:
            print(f"  [生成] {output_file.relative_to(ROOT)}")
        # 若源文件就是 optimizable 文件本身（已提取的情况），跳过提取
        if source_file != output_file and source_file.exists():
            func_source = _extract_function(source_file, func_name)
            module_content = _build_optimizable_module(
                func_name, func_source, description, entry["source_file"]
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(module_content, encoding="utf-8")
        else:
            print(f"  [警告] source_file={entry['source_file']} 不存在，跳过生成", file=sys.stderr)
    else:
        if verbose:
            print(f"  [已存在] {output_file.relative_to(ROOT)}")

    # ── Step 2: 检查 source_file 是否已接管 import ────────────────────────────
    if source_file.exists() and source_file != output_file:
        source_text = source_file.read_text(encoding="utf-8")
        # 计算 Python 模块路径：src/optimizable/xxx.py → src.optimizable.xxx
        rel_output   = output_file.relative_to(ROOT)
        module_path  = str(rel_output.with_suffix("")).replace("/", ".")

        if not _is_import_present(source_text, module_path, func_name):
            print(
                f"\n  [提示] {entry['source_file']} 尚未从 optimizable 模块导入 {func_name}。\n"
                f"  请手动在文件顶部添加：\n"
                f"    from {module_path} import {func_name}\n"
                f"  并将原调用点替换为对 {func_name}() 的调用。\n"
                f"  参考已有案例：src/optimizer/multiadam.py",
                file=sys.stderr
            )
        else:
            if verbose:
                print(f"  [已接管] {entry['source_file']} ← {func_name}()")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  子命令实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_list(_args) -> None:
    """列出所有候选函数。"""
    cfg = _load_config()

    print("\n━━━━━━━━━━━━━━━━ 已注册（将被 ReEvo2D 优化）━━━━━━━━━━━━━━━━")
    for e in cfg.get("registered", []):
        output_exists = (ROOT / e["output_file"]).exists()
        status = "✓" if output_exists else "✗ (文件缺失)"
        print(f"  [{status}] {e['function']:30s}  ←  {e['source_file']}")
        print(f"             描述: {e.get('description', '')[:80]}")
        scenarios = e.get("eval_scenarios", [])
        if scenarios:
            sc_str = ", ".join(f"{s['pde']}({s.get('method','adam')})" for s in scenarios)
            print(f"             评估: {sc_str}")
        print()

    print("\n━━━━━━━━━━━━━━━━ 候选函数（可添加到 registered）━━━━━━━━━━━━━━")
    for e in cfg.get("candidates", []):
        src_path = ROOT / e["source_file"]
        available = src_path.exists()
        status = "可用" if available else "文件不存在"
        print(f"  [{status}] {e['function']:30s}  ←  {e['source_file']}")
        print(f"             描述: {e.get('description', '')[:80]}")
        suggested = e.get("suggested_output", "")
        if suggested:
            print(f"             输出: src/optimizable/{suggested}.py")
        print()

    print("提示: 运行 --add --source <文件> --function <函数名> 激活候选项")


def cmd_status(_args) -> None:
    """显示已注册函数的详细状态。"""
    cfg = _load_config()
    print("\n已注册函数状态：\n")

    for e in cfg.get("registered", []):
        func      = e["function"]
        src_path  = ROOT / e["source_file"]
        out_path  = ROOT / e["output_file"]

        file_ok   = "✓" if out_path.exists() else "✗"
        src_ok    = "✓" if src_path.exists() else "N/A"

        import_ok = "?"
        if src_path.exists() and src_path != out_path:
            rel_output  = out_path.relative_to(ROOT)
            module_path = str(rel_output.with_suffix("")).replace("/", ".")
            src_text    = src_path.read_text(encoding="utf-8")
            import_ok   = "✓" if _is_import_present(src_text, module_path, func) else "✗"
        elif src_path == out_path:
            import_ok = "N/A(直接文件)"

        print(f"  {func}")
        print(f"    optimizable 文件: {file_ok}  {e['output_file']}")
        print(f"    源文件存在:       {src_ok}  {e['source_file']}")
        print(f"    import 已注入:    {import_ok}")
        print()


def cmd_apply(_args) -> None:
    """应用所有已注册的条目（幂等）。"""
    cfg = _load_config()
    registered = cfg.get("registered", [])

    if not registered:
        print("registered 为空，请先在 optimizable_config.yaml 中添加条目。")
        return

    print(f"正在注册 {len(registered)} 个函数...\n")
    for entry in registered:
        print(f"→ {entry['function']}  ({entry['source_file']})")
        register_one(entry, verbose=True)

    print("\n[完成] 所有函数已注册。")
    print("提示：若提示需手动注入 import，请参考 src/optimizer/multiadam.py 的改造示例。")


def cmd_add(args) -> None:
    """将新函数添加到 registered 并立即 apply。"""
    cfg = _load_config()

    output_name = args.output or f"{args.function}_optimizable"
    if not output_name.endswith("_optimizable"):
        output_name = output_name + "_optimizable"
    output_file = f"src/optimizable/{output_name}.py"

    # 检查是否已存在
    for e in cfg["registered"]:
        if e["function"] == args.function and e["source_file"] == args.source:
            print(f"[已存在] {args.function} 已在 registered 中，跳过添加。")
            cmd_apply(args)
            return

    # 解析评估场景
    pdes    = args.eval_pde    or ["Burgers1D"]
    methods = args.eval_method or ["adam"]
    iters   = args.eval_iter   or [5000]

    # 对齐长度
    max_len = max(len(pdes), len(methods), len(iters))
    pdes    = (pdes    * max_len)[:max_len]
    methods = (methods * max_len)[:max_len]
    iters   = (iters   * max_len)[:max_len]

    eval_scenarios = [
        {"pde": p, "method": m, "iter": i, "weight": 1.0}
        for p, m, i in zip(pdes, methods, iters)
    ]

    new_entry = {
        "source_file":    args.source,
        "function":       args.function,
        "output_file":    output_file,
        "description":    args.desc or args.function,
        "eval_scenarios": eval_scenarios,
    }

    cfg["registered"].append(new_entry)
    _save_config(cfg)
    print(f"[已添加] {args.function} → {output_file}")

    register_one(new_entry, verbose=True)


def cmd_remove(args) -> None:
    """从 registered 中注销函数（不删除生成文件）。"""
    cfg = _load_config()
    before = len(cfg["registered"])
    cfg["registered"] = [
        e for e in cfg["registered"]
        if e["function"] != args.function
    ]
    after = len(cfg["registered"])

    if before == after:
        print(f"[未找到] '{args.function}' 不在 registered 中")
        return

    _save_config(cfg)
    print(f"[已注销] '{args.function}' 已从 registered 移除（生成文件未删除）")
    print(f"提示：若需还原调用路径，手动删除对应 src/optimizable/*.py 中的 import 语句")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="管理 PINNacle × ReEvo2D 的可优化函数注册",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",   action="store_true", help="列出所有候选和已注册函数")
    group.add_argument("--status", action="store_true", help="检查当前注册状态")
    group.add_argument("--apply",  action="store_true", help="应用 yaml 中的 registered 配置")
    group.add_argument("--add",    action="store_true", help="添加新函数到 registered")
    group.add_argument("--remove", action="store_true", help="从 registered 注销函数")

    # --add 参数
    parser.add_argument("--source",      default=None, help="源文件路径（相对于项目根目录）")
    parser.add_argument("--function",    default=None, help="函数名")
    parser.add_argument("--output",      default=None, help="输出文件名（不含 .py），默认 {function}_optimizable")
    parser.add_argument("--desc",        default=None, help="函数描述（供 LLM 理解优化目标）")
    parser.add_argument("--eval-pde",    nargs="+",    help="评估 PDE 名称列表")
    parser.add_argument("--eval-method", nargs="+",    help="训练方法列表（adam/multiadam/ntk/lra/rar）")
    parser.add_argument("--eval-iter",   nargs="+", type=int, help="各场景迭代步数")

    # --remove 参数（共用 --function）
    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    elif args.status:
        cmd_status(args)
    elif args.apply:
        cmd_apply(args)
    elif args.add:
        if not args.source or not args.function:
            parser.error("--add 需要 --source 和 --function 参数")
        cmd_add(args)
    elif args.remove:
        if not args.function:
            parser.error("--remove 需要 --function 参数")
        cmd_remove(args)


if __name__ == "__main__":
    main()
