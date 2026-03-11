# PINNacle × ReEvo2D 使用指南

> 本指南面向希望使用 LLM 进化优化器自动改进 PINN 算法组件的研究者和工程师。

---

## 目录

1. [整体架构](#1-整体架构)
2. [快速开始](#2-快速开始)
3. [选择优化目标](#3-选择优化目标——最重要的决策)
   - [3b. 多 PDE 联合评估](#3b-多-pde-联合评估)
4. [注册新的优化目标](#4-注册新的优化目标)
5. [配置评估场景](#5-配置评估场景)
6. [运行优化器](#6-运行优化器)
7. [解读结果](#7-解读结果)
8. [进阶配置](#8-进阶配置)
9. [FAQ](#9-faq)

---

## 1. 整体架构

```
你的决策层
  ↓ 在 optimizable_config.yaml 中选择"优化哪些函数"
  ↓ 运行 scripts/register_optimizable.py --apply 生成接管代码

ReEvo2D 进化层（optimizer/reevo2d.py）
  ↓ 调用 get_funcs() 读取可优化函数
  ↓ 让 LLM 生成新的函数实现
  ↓ 注入 src/optimizable/*.py

评估层（scripts/run_experiment.py）
  ↓ 子进程运行 benchmark_single.py（单 PDE）
  ↓ 解析 "Final L2RE: <float>" 指标
  ↑ 返回 L2RE 给 ReEvo2D 作为适应度

LLM 反思层
  ↑ 比较 better/worse 代码 → 短期反思
  ↑ 积累历史反思 → 长期反思
  ↓ 指导下一轮变异/交叉
```

**数据流总结：** LLM 写代码 → 注入文件 → 训练 PINN → 测 L2RE → LLM 反思 → 循环

---

## 2. 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install pyyaml deepxde   # 额外需要

# 2. 设置 LLM API Key
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://api.example.com/v1  # 如使用代理（可选）
```

### 一键启动（使用默认配置）

```bash
python main.py --model deepseek-coder --epochs 5000 --max-fe 200
```

这会用当前注册的 10 个优化目标（四大方向全覆盖），
通过 `global_eval_scenarios` 在 8 个 PDE × 方法组合上联合评估，跑 200 轮进化。

### 查看进度

```bash
# 实时查看 ReEvo2D 日志
tail -f runs/pinnacle_reevo/problems/pinnacle/*.txt

# 查看当前最优函数
ls runs/pinnacle_reevo/problems/pinnacle/
```

---

## 3. 选择优化目标——最重要的决策

**选择标准（优先考虑以下三点）：**

| 标准 | 说明 |
|------|------|
| **函数独立性** | 函数是纯计算逻辑，不依赖全局状态，签名清晰 |
| **影响 L2RE 的直接性** | 改函数 → 直接影响训练 → 影响测试误差 |
| **LLM 可合理探索的空间** | 有多种合理的备选实现（不是只有一个正确答案） |

### 查看所有候选函数

```bash
python scripts/register_optimizable.py --list
```

输出示例：
```
━━━━━━━━━━━━━━━━ 已注册（将被 ReEvo2D 优化）━━━━━━━━━━━━━━━━

  [✓] compute_group_update          ←  src/optimizable/multiadam_optimizable.py
       描述: MultiAdam 多损失组梯度融合：将各损失组的 Adam 更新量加权合并为最终参数更新
       评估: Burgers1D(multiadam), Poisson2D_Classic(multiadam)

━━━━━━━━━━━━━━━━ 候选函数（可添加到 registered）━━━━━━━━━━━━━━

  [可用] sadam                      ←  src/optimizer/multiadam.py
       描述: [高价值] 完整 MultiAdam 函数式 API：包含每参数的一阶/二阶矩更新...
       输出: src/optimizable/multiadam_sadam_optimizable.py
```

### 四个方向覆盖总览

PINN 求解 PDE 的四个主流研究方向，当前优化目标覆盖情况：

| 方向 | 覆盖率 | 已注册函数 |
|------|--------|-----------|
| **① 网络架构** | ~95% | `encode_input`, `fnn_forward_body`, `laaf_scale`, `apply_ic_decay` |
| **② 训练优化** | ~90% | `compute_group_update`, `adapt_loss_weights`, `adapt_boundary_weight`, `select_optimizer` |
| **③ 采样策略** | ~80% | `select_resample_points`, `compute_causal_weights` |
| **④ 损失设计** | ~80% | `adapt_loss_weights`, `adapt_boundary_weight`, `compute_causal_weights` |

### 已注册函数一览（共 10 个）

#### 方向① 网络架构层

| 函数名 | 优化价值 | 核心探索方向 |
|--------|----------|-------------|
| `encode_input` | ⭐⭐⭐ | Random Fourier Features、NERF 位置编码、多尺度特征拼接（缓解谱偏置） |
| `fnn_forward_body` | ⭐⭐⭐ | 残差连接、Highway 网络、Modified MLP（Wang et al. 2022）、注意力门控 |
| `apply_ic_decay` | ⭐⭐⭐ | sigmoid/tanh/多项式衰减、空间自适应混合权重 |
| `laaf_scale` | ⭐⭐ | softplus(a) 有界缩放、归一化缩放、layer-wise vs element-wise |

#### 方向② 训练优化层

| 函数名 | 优化价值 | 核心探索方向 |
|--------|----------|-------------|
| `compute_group_update` | ⭐⭐⭐ | 梯度投影（PCGrad）、归一化融合、动态门控权重 |
| `adapt_loss_weights` | ⭐⭐⭐ | EMA 平滑、对数尺度、分层 NTK 适配 |
| `adapt_boundary_weight` | ⭐⭐ | 自适应 alpha、权重裁剪、Trust Region 更新 |
| `select_optimizer` | ⭐⭐ | 损失平台检测切换、梯度范数触发、软切换概率 |

#### 方向③ 采样策略层 / 方向④ 损失设计层

| 函数名 | 方向 | 优化价值 | 核心探索方向 |
|--------|------|----------|-------------|
| `select_resample_points` | ③ 采样 | ⭐⭐⭐ | 概率采样、多样性约束、课程式自适应 |
| `compute_causal_weights` | ③④ 共用 | ⭐⭐⭐ | Wang et al. 2022 因果权重、软因果、自适应 epsilon、反向课程 |

> `compute_causal_weights` 同时作用于采样权重（③）和损失加权（④），
> 对时变 PDE（Burgers、Wave、Heat 时变类）影响最大，稳态 PDE 中不会被调用。

### 候选函数（可激活）

| 函数名 | 来源 | 优化价值 | 探索方向 |
|--------|------|----------|----------|
| `sadam` | `src/optimizer/multiadam.py` | ⭐⭐⭐ | AMSGrad、梯度裁剪、动量聚合 |
| `use_gepinn` | `src/pde/baseclass.py` | ⭐⭐ | 高阶导数、曲率正则、选择性增强 |
| `random_points` | `src/utils/geom.py` | ⭐⭐ | QMC、层次采样、贴边增强 |
| `gaussian_random_field` | `src/utils/random.py` | ⭐⭐ | 自适应功率谱、各向异性场 |

---

## 3b. 多 PDE 联合评估

这是推荐的评估模式：所有注册函数在同一套 PDE 场景上**一起**评估，共享同一套代码注入。

### 配置方式

在 `optimizable_config.yaml` 中填写 `global_eval_scenarios`（已预配置 8 个场景）：

```yaml
global_eval_scenarios:
  - {pde: Burgers1D,            method: multiadam, ...}  # compute_group_update, fnn_forward_body, encode_input, compute_causal_weights
  - {pde: Poisson2D_Classic,    method: ntk,       ...}  # adapt_loss_weights, fnn_forward_body, encode_input
  - {pde: Heat2D_Multiscale,    method: lra,       ...}  # adapt_boundary_weight, fnn_forward_body, encode_input（多尺度谱偏置）
  - {pde: Burgers1D,            method: rar,       ...}  # select_resample_points, fnn_forward_body
  - {pde: Wave1D,               method: adam,      ...}  # apply_ic_decay, compute_causal_weights（强时序）
  - {pde: Burgers1D,            method: laaf,      ...}  # laaf_scale
  - {pde: Poisson2D_Classic,    method: lbfgs,     ...}  # select_optimizer
  - {pde: Wave2D_Heterogeneous, method: adam,      ...}  # encode_input（高频波动，谱偏置最典型）
```

### 关键机制

**各函数在 8 个场景中的执行矩阵（✓ = 实际调用 / ✗ = 注入但未执行）：**

| 场景 | `encode_input` | `fnn_forward_body` | `apply_ic_decay` | `compute_causal_weights` | `laaf_scale` | `compute_group_update` | `adapt_loss_weights` | `select_resample_points` | `adapt_boundary_weight` | `select_optimizer` |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Burgers1D + multiadam | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Poisson2D + ntk       | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Heat2D + lra          | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Burgers1D + rar       | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Wave1D + adam         | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Burgers1D + laaf      | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Poisson2D + lbfgs     | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Wave2D_Het + adam     | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

每个函数至少在 1 个场景中被真正调用，均能获得有效进化信号。

- ✓ 执行的场景会直接反映在 L2RE 上，作为该函数的有效评估信号
- ✗ 未调用的场景，该函数代码被注入但从未运行，L2RE 由其他已执行函数决定
- **不存在"冲突"**：各函数的代码路径在运行时互相独立

### 适应度聚合

最终适应度 = Σ (L2RE_i × weight_i) / Σ weight_i

只要 `global_eval_scenarios` 中每类函数至少有一个 PDE 场景真正调用它，
该函数就能获得有效的进化信号。未被任何场景调用的函数相当于"搭便车"——不贡献正向信号，也不造成干扰。

---

## 4. 注册新的优化目标

### 方式 A：编辑 `optimizable_config.yaml`（推荐）

打开 `optimizable_config.yaml`，将 `candidates` 中你想激活的条目复制到 `registered`，
然后运行：

```bash
python scripts/register_optimizable.py --apply
```

**示例**：激活 `sadam` 函数

```yaml
# optimizable_config.yaml
registered:
  # ... 已有条目 ...

  - source_file:  src/optimizer/multiadam.py      # 原始源文件
    function:     sadam                            # 要优化的函数名
    output_file:  src/optimizable/multiadam_sadam_optimizable.py  # 生成的包装文件
    description:  "完整 MultiAdam 参数更新循环"
    eval_scenarios:
      - {pde: Burgers1D,         method: multiadam, iter: 5000, weight: 1.0}
      - {pde: Heat2D_Multiscale, method: multiadam, iter: 5000, weight: 1.0}
```

### 方式 B：命令行直接添加

```bash
python scripts/register_optimizable.py --add \
    --source src/optimizer/multiadam.py \
    --function sadam \
    --output multiadam_sadam_optimizable \
    --desc "完整 MultiAdam 参数更新循环（含动量维护）" \
    --eval-pde Burgers1D Poisson2D_Classic \
    --eval-method multiadam multiadam \
    --eval-iter 5000 5000
```

### 方式 C：完全自定义函数

如果你想优化某个目前不在候选列表中的函数（例如你自己写的 PDE 定义）：

1. **确认函数是否适合优化**（见第 3 节标准）
2. 直接在 `optimizable_config.yaml` 的 `registered` 中添加条目
3. 若函数在已有 `*_optimizable.py` 中，设置 `source_file` = `output_file`
4. 若函数在原始源文件中，`--apply` 会自动提取并生成包装文件，但需要手动在原文件中添加 import

### 注册后的必要步骤

`--apply` 会生成 `src/optimizable/*.py` 文件。对于从原始源文件提取的函数，
你还需要在源文件中将原调用替换为 import：

```python
# 在源文件顶部添加（register_optimizable.py 会提示你）
from src.optimizable.your_new_optimizable import your_function

# 在原调用点，将内联逻辑替换为函数调用：
result = your_function(arg1, arg2)  # [OPTIMIZABLE]
```

参考已有示例：`src/optimizer/multiadam.py` 第 104 行附近的改造方式。

### 注销优化目标

```bash
python scripts/register_optimizable.py --remove --function sadam
```

注：这只会从 `registered` 配置中移除，不会删除已生成的文件或还原调用路径。

---

## 5. 配置评估场景

评估场景决定"用什么 PDE、什么方法、训练多少步来判断函数好坏"。

### 两种配置方式及优先级

**方式一：`global_eval_scenarios`（推荐，全局生效）**

所有注册函数在同一套场景上联合评估，同时注入、同时测试：

```yaml
global_eval_scenarios:           # 顶级字段，非空时优先于各函数的 eval_scenarios
  - {pde: Burgers1D, method: multiadam, iter: 5000, weight: 1.0}
  - {pde: Wave1D,    method: adam,      iter: 5000, weight: 0.8}
  # ...
```

**方式二：`eval_scenarios`（逐函数独立评估）**

当 `global_eval_scenarios` 为空列表时生效，每个函数用自己的场景单独评估：

```yaml
registered:
  - function: compute_group_update
    eval_scenarios:
      - {pde: Burgers1D,         method: multiadam, iter: 5000, weight: 1.0}
      - {pde: Poisson2D_Classic, method: multiadam, iter: 5000, weight: 1.5}
      # weight 决定此场景在总适应度中的权重
```

> **选哪种？** 优化目标之间有相互作用（如 `encode_input` 影响所有方法）时用 global；
> 只想单独评估某一个函数时用 per-function。

### 支持的 PDE 名称

| 类别 | PDE 名称 |
|------|----------|
| Burgers | `Burgers1D`, `Burgers2D` |
| Poisson | `Poisson2D_Classic`, `PoissonBoltzmann2D`, `Poisson3D_ComplexGeometry`, `Poisson2D_ManyArea` |
| Heat | `Heat2D_VaryingCoef`, `Heat2D_Multiscale`, `Heat2D_ComplexGeometry`, `Heat2D_LongTime` |
| Navier-Stokes | `NS2D_LidDriven`, `NS2D_BackStep`, `NS2D_LongTime` |
| Wave | `Wave1D`, `Wave2D_Heterogeneous`, `Wave2D_LongTime` |
| Chaotic | `KuramotoSivashinskyEquation`, `GrayScottEquation` |
| High-dim | `PoissonND`, `HeatND` |

### 支持的训练方法（--method）

| 方法名 | 对应算法 |
|--------|----------|
| `adam` | 标准 Adam |
| `multiadam` | MultiAdam（多损失组） |
| `ntk` | NTK 自适应权重 |
| `lra` | 学习率退火（LRA） |
| `lbfgs` | Adam + L-BFGS 切换 |
| `rar` | 残差自适应重采样 |
| `gepinn` | 梯度增强 PINN |
| `laaf` | 局部自适应激活函数 |
| `gaaf` | 全局自适应激活函数 |

### 评估场景选择建议

- **迭代步数（iter）**：5000-10000 步是效率与精度的平衡点。太少 (< 2000) 噪声大，太多 (> 20000) 每次评估太慢
- **多 PDE 场景**：至少选 2 个不同类型的 PDE，避免过拟合单一问题
- **方法匹配**：`compute_group_update` 用 `multiadam` 方法评估，`select_resample_points` 用 `rar` 方法评估

---

## 6. 运行优化器

### 基础启动

```bash
export OPENAI_API_KEY=your_key
python main.py --model deepseek-coder
```

### 完整参数

```bash
python main.py \
    --model         deepseek-coder \   # LLM 模型（见下方支持列表）
    --api-key       $OPENAI_API_KEY \  # 也可用环境变量
    --base-url      https://... \      # 代理 URL（可选）
    --temperature   0.5 \              # LLM 采样温度
    --max-fe        200 \              # 最大函数评估次数（主停止条件）
    --pop-size      10 \               # 每代种群大小
    --init-pop-size 20 \               # 初始种群大小（多样化初始化）
    --mutation-rate 0.8 \              # 变异率（相对于交叉）
    --epochs        5000 \             # 每次评估的训练迭代步数
    --output-dir    ./runs/exp1        # 结果输出目录
```

### 支持的 LLM 模型

| 模型 | `--model` 值 | 备注 |
|------|-------------|------|
| GPT-4o | `gpt-4o` | 效果最好，成本最高 |
| GPT-4o-mini | `gpt-4o-mini` | 成本低，推荐入门 |
| DeepSeek-Coder | `deepseek-coder` | 代码能力强，性价比高 |
| GLM-4 | `GLM-4` | 国内访问友好 |
| LlamaAPI 系列 | `llama-3.1-70b` | 开源模型 |

### 估算运行时间

```
总时间 ≈ max_fe × 每次评估时间

每次评估时间（单 PDE × 5000 步）：
  CPU：1-3 分钟 / GPU：20-40 秒

使用 global_eval_scenarios（默认 8 个场景）：
  CPU：8-24 分钟/评估，max_fe=200 约 27-80 小时
  GPU：3-5 分钟/评估，max_fe=200 约 10-17 小时

减少场景数量可线性缩短时间：保留 3 个关键场景 → GPU 约 3-5 小时
```

---

## 7. 解读结果

### 输出目录结构

```
runs/pinnacle_reevo/
├── problems/pinnacle/
│   ├── problem_iter0_code0.txt          # 初始种群代码
│   ├── problem_iter0_stdout0.txt        # 对应训练输出
│   ├── problem_iter1_response0.txt      # LLM 生成的新实现
│   ├── problem_iter1_short_term_reflections.txt   # LLM 短期反思
│   ├── problem_iter1_long_term_reflection.txt     # LLM 长期反思（累积知识）
│   └── ...
└── (日志文件)
```

### 日志中的关键信息

```
# 每次评估的结果
Iteration 5: API eval obj=0.0432, scores=[{'name': 'L2RE_Burgers1D_multiadam', 'value': 0.0432, ...}]

# 精英更新
Iteration 5: Elitist: 0.0312   ← 当前最优适应度

# 完成
Best code path: .../problem_iter42_funcIndex0_code3.txt
Best code:
def compute_group_update(exp_avg, denom, group_weights):
    ...  ← LLM 找到的最优实现
```

### 如何使用最优结果

1. 找到 `Best code path` 指向的文件
2. 将其中的函数实现复制到 `src/optimizable/` 对应文件中
3. 正常使用 `benchmark.py` 跑完整 20 案例评估

---

## 8. 进阶配置

### 只优化特定函数（不是全部 registered）

修改 `main.py` 中的 `reevo_func_index`：

```python
# main.py 中的 _build_cfg()
cfg = SimpleNamespace(
    ...
    reevo_func_index=[0, 2],  # 只优化 get_funcs() 返回列表中索引为 0 和 2 的函数
    algorithm="reevo2d",      # "reevo2d" 优化所有, 其他值只优化 reevo_func_index
)
```

### 添加外部知识（Domain Knowledge）

在 `src/optimizable/*.py` 的 docstring 中添加 `[外部知识]` 区块，ReEvo2D 会将其注入 LLM prompt：

```python
def compute_group_update(exp_avg, denom, group_weights):
    """
    [MULTIADAM-CORE] 融合各损失组的 Adam 更新...

    [外部知识]
    - 研究发现 PCGrad（2020）通过投影去除冲突梯度能提升多任务学习性能
    - PINN 梯度冲突主要发生在 PDE 残差与边界条件之间
    - 对高刚性 PDE（如 Helmholtz），BC 梯度往往数量级更大
    """
```

### 自定义 LLM prompt

所有 prompt 模板在 `prompts/common/` 目录，可直接编辑：

| 文件 | 作用 |
|------|------|
| `system_generator.txt` | 生成器角色设定 |
| `system_reflector.txt` | 反思器角色设定 |
| `user_generator.txt` | 生成新函数的指令格式 |
| `seed.txt` | 初始种子函数提示 |
| `user_reflector_st.txt` | 短期反思（比较两段代码） |
| `user_reflector_lt.txt` | 长期反思（积累知识） |
| `crossover.txt` | 交叉操作提示 |
| `mutation.txt` | 变异操作提示 |

### 直接运行单场景实验（不启动 ReEvo2D）

```bash
# 手动测试某个 PDE + 方法组合
python benchmark_single.py --pde Burgers1D --iter 10000 --method multiadam

# 手动测试函数注入效果
python scripts/run_experiment.py \
    --spec my_replacement.json \
    --epochs 5000 \
    --pretty \
    --no-stdout
```

`my_replacement.json` 格式：
```json
[{
  "function": "compute_group_update",
  "file": "multiadam_optimizable.py",
  "code": "def compute_group_update(exp_avg, denom, group_weights):\n    ..."
}]
```

---

## 9. FAQ

**Q: 我想优化自己写的 PDE 的某个部分，怎么做？**

A: 把该部分逻辑提取为一个独立函数（确保签名清晰、无副作用），然后用 `--add` 注册。
重点是：函数的输入输出要定义好，不要依赖 `self` 或全局状态。

---

**Q: LLM 产出的代码报错，进化会中断吗？**

A: 不会。ReEvo2D 将报错个体标记为无效（obj=inf），继续进化其他个体。
错误信息会被反馈给 LLM 用于反思，帮助下一轮生成更正确的代码。

---

**Q: 如何控制评估速度和成本之间的权衡？**

A: 三个旋钮：
- `--epochs`: 越小评估越快，但 L2RE 估计噪声越大（推荐 3000-8000）
- `--pop-size`: 越小每代计算量越小（推荐 5-15）
- `eval_scenarios` 中的 PDE 数量：越少越快，但泛化性评估不足（推荐 2-3 个）

---

**Q: 为什么要同时配置多个评估 PDE？**

A: 防止 LLM "过拟合"单一 PDE。一个只对 Burgers1D 有效的技巧，
在 Poisson2D 上可能完全无效（甚至变差）。多场景加权评估迫使 LLM 寻找通用改进。

---

**Q: `register_optimizable.py --apply` 报错"函数未找到"怎么办？**

A: 检查 `optimizable_config.yaml` 中的 `function` 字段是否与源文件中的函数名精确匹配（大小写敏感）。
运行 `python scripts/register_optimizable.py --list` 查看实际可用的函数名。

---

**Q: 进化结束后发现结果不好，如何继续？**

A: 目前 ReEvo2D 不支持断点续跑（每次从头开始），但可以：
1. 手动将上次找到的最优函数复制到 `src/optimizable/*.py` 作为新的 seed
2. 适当增大 `--temperature` 增加多样性
3. 增大 `--init-pop-size` 让初始种群更丰富
