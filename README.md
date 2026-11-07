# AMA 564 课程项目：一致性门的作用点在步长上

TENT 在混合在线分布漂移下的实证研究

## 结论摘要

本项目检验了"按样本筛选梯度能提升 TENT 在混合在线漂移下的精度"这一假设。结果未支持该假设：TENT 的主要增益来自 BN
统计量重估；一致性门控的有效部分实际作用于每批更新量，等位移配平后仅余 +0.16 ± 0.13 pp。我们进一步定位了失效边界：
在线流 batch 16 上放大更新量的方法显著低于冻结模型，`jpeg_compression` 上的损失来自统计量替换本身。方法的代价是单步
延迟为 TENT 的 2.0 倍。

以下读数限于这份 checkpoint、这批格子和这三个 seed。本项目的主要产出是一组诊断结果：筛选机制的实际作用点落在每批
更新量上，其失效边界与代价也已定位；方法在绝对精度上的领先是次要的。

## 1. 研究问题与事先固定的判定标准

**问题：** 在混合、在线分布漂移下，给 TENT 加"按样本筛选梯度"的目标后精度如何变化；test-time adaptation 相对冻结源
模型失去优势的边界在哪里。

**原预期：** TENT 对每个样本统一做熵最小化，会在已判错样本上放大误差；用熵阈值门和增广一致性门筛出值得更新的样本，
应能提高精度。

**事先固定的判定标准（preregistration）：** `configs/pilot.yaml` 要求同时满足：

- TENT 在 10 个验证格上平均比源模型高 ≥1.0 pp；
- 任何一种腐蚀的跌幅不超过 0.5 pp。

**实测：** 第一条满足，TENT 相对源模型 +10.97 pp；第二条被 `jpeg_compression` 的平均 −2.20 pp 否定。
`run.runAll pilot` 打印 `FAIL -> switch to plan B`。同一张表里 `bnAdapt` 的均值增益是 +10.95 pp，说明这一段绝大部分
在只重估 BN 统计量、不执行任何梯度步的方法上同样成立。

研究因此按预设分支改问：筛选实际改变了什么。读数是每批的更新量。

## 2. 实验设置

**源模型：** torchvision ResNet-18（11,181,642 个参数），CIFAR-10 从零训练 100 epoch，测试精度 88.07%；六个方法共用
同一冻结 checkpoint。

**可更新参数：** 仅 40 个 BN 张量中的 9,600 个 scale/shift；卷积与全连接权重全程冻结。

**数据面：** CIFAR-10-C。验证格 10 个（seen 5 种腐蚀 × severity 3/5，每格 5,000 张，永远不进结论）；测试格 65 个
（seen 15 + unseen 50，每格 10,000 张）；在线流 25,000 张，5 种未见腐蚀 severity 5 顺序拼接，段间不重置；batch
16/64/256；3 seeds。

**协议：** 静态格逐格重载冻结 checkpoint；在线流不重置；train-mode BN；每批一步；更新后用更新后的 BN 仿参对同一批取
logits；共享批顺序并补齐满 batch。

**六方法：**

| 方法 | 更新目标 | 读标签 |
|---|---|---|
| `sourceOnly` | 不更新，BN 用源训练的 running 统计量 | 否 |
| `bnAdapt` | 不更新权重，只让 BN 使用当前 batch 统计量 | 否 |
| `tent` | 平均预测熵，lr 0.001 | 否 |
| `tentTuned` | 同一目标，lr 0.064（验证格选出） | 否 |
| `consistencyTent` | `Σ w_i (H_i + 16·d_i) / Σ w_i`，λ=16 | 否 |
| `oracle` | 平均预测熵，只计入更新前已预测正确的样本 | 是，作为上限参照 |

**指标：** 主指标为每格分类精度，聚合为 65 格均值、seen/unseen 分组均值、按 severity 分组均值，均给三个 seed 的均值±
标准差。辅助指标为相对错误降低率、同位移处精度差、最差格/最差段、每批延迟、峰值内存、可更新参数量。

**调参预算对等：** 两侧都在同一批 10 个验证格上以完整 pass 计数，并跑到曲线掉头；λ=16 与 lr=0.064 都是两侧有实测点的
内点。

**位移：** 每行记录该格 9,600 个 BN 仿参更新前后的欧氏距离，用实测 `displacementNorm` 替代 `lr × ‖∇L‖ × 每格批数`
的估计值。

## 3. 主要结果

### 3.1 增益分成统计量重估与熵目标

10 个验证格：`sourceOnly` 67.12%、`bnAdapt` 78.06%、`tent` 78.09%。从 67% 到 78% 几乎全部来自 BN 改用当前 batch
统计量；熵目标在 `bnAdapt` 之上增加 0.02 pp，十个格子里六个为正。

65 个测试格：`sourceOnly` 76.45%、`bnAdapt` 79.97%、`tent` 80.08%。统计量重估 +3.53 pp，熵目标再 +0.11 pp；seen 15
格 +0.09 pp，unseen 50 格 +0.11 pp。两个格子集合给出的比例一致：lr 0.001 的那一步几乎全部叠在统计量重估之上。

`jpeg_compression` 是唯一冻结模型高于全部适应方法的腐蚀：`sourceOnly` 84.21%、`bnAdapt` 81.53%、`tent` 81.60%、
`consistencyTent` 83.14%。损失出在"用 batch 统计量替换 running 统计量"这一步本身，两道门碰不到它；这是方法的结构性
边界。

### 3.2 有效增益来自一致性损失项，峰值在 λ=16

同一批 10 个验证格、lr 0.001、batch 64、三个 seed。基线 `tent` 为 77.90%。

| 变体 | 配置 | 相对 TENT | 一格位移 |
|---|---|---|---|
| `entropyGateOnly_q75` | 熵门保留 75%，无一致性项 | −0.01 pp | 0.011 |
| `entropyGateOnly_q50` | 熵门保留 50%，无一致性项 | −0.03 pp | 0.003 |
| `consistencyWeightOnly` | 一致性分数只加权样本（λ=1） | +0.01 pp | 0.017 |
| `consistencyLossOnly` | 一致性分数只作损失项（λ=1） | +0.23 pp | 0.058 |
| `bothGates_lambda4` | 两个角色都启用（λ=4） | +0.59 pp | 0.146 |
| `bothGates_lambda8` | 两个角色都启用（λ=8） | +0.91 pp | 0.267 |
| `bothGates_lambda16` | 两个角色都启用（λ=16，冻结值） | **+1.26 pp** | 0.473 |
| `bothGates_lambda32` | 两个角色都启用（λ=32） | +1.23 pp | 0.807 |
| `bothGates_lambda64` | 两个角色都启用（λ=64） | +0.16 pp | 1.314 |
| `entropyGate_q75` | 熵门 q75 + λ=16 | +0.84 pp | 0.393 |
| `entropyGate_q50` | 熵门 q50 + λ=16 | +0.67 pp | 0.356 |

结论：

- 可检出的增益都在一致性损失项；λ=16 是内点峰值，两侧都有实测读数（λ=1 与 2 时是 +0.19 与 +0.33 pp），λ=64
  的回落并不平滑。
- 熵门单独使用为负；同一道门叠在 λ=16 上把 +1.26 pp 削成 +0.84 pp 与 +0.67 pp，付出的是样本量，方向信息没有改变。
- 按分歧大小重排样本权重的 +0.01 pp 在噪声内（10 格配对差为 5 正、2 负、3 持平）。
- 最终配置为 `entropyGate.use: false`、`consistencyGate.weight: 16.0`；一致性项两个角色留在配置中，实测有效的是
  损失项。

### 3.3 位移解释增益主体，等位移处余 +0.16 pp

两条阶梯放在同一批验证格、batch 64、三个 seed：

- TENT 沿步长：lr 取 0.001 的 0.5–128 倍，每格位移从 0.009 走到 1.517；精度从 77.88 升到 lr 0.064 处的峰值 78.95，
  再加一档回落到 77.68。
- 一致性项只作损失项：λ=1/4/8/16/32，位移 0.058 → 1.075，精度 78.13 / 78.62 / 78.97 / 79.15 / 78.53，最高在 λ=16。

λ=16 用更短的一格位移（0.636 对 0.850）取得更高精度。按每个 seed 自己的 TENT 阶梯插值到同位移：

| λ | 每格位移 | 精度 (%) | TENT 同位移处 (%) | 差 (pp) |
|---|---|---|---|---|
| 1 | 0.058 | 78.13 | 78.03 | +0.10 |
| 4 | 0.199 | 78.62 | 78.42 | +0.20 |
| 8 | 0.363 | 78.97 | 78.73 | +0.24 |
| 16 | 0.636 | 79.15 | 78.91 | +0.24 |
| 32 | 1.075 | 78.53 | 78.43 | +0.10 |

18 个读数的均值为 +0.16 ± 0.13 pp，其中 16 个为正。λ=16 相对 lr 0.001 的 TENT 高 1.25 pp，其中约五分之一（+0.24 pp）
出现在位移相同处，其余部分与单纯把步长放大所得的相同。

### 3.4 65 格主表：consistencyTent 在四个口径上最高

| 方法 | 65 格 (%) | seen 15 (%) | unseen 50 (%) | unseen sev5 (%) | 一格位移 |
|---|---|---|---|---|---|
| `sourceOnly` | 76.45 | 78.10 | 75.95 | 64.72 | 0.000 |
| `bnAdapt` | 79.97 | 81.65 | 79.47 | 75.03 | 0.000 |
| `tent` | 80.08 | 81.74 | 79.58 | 75.18 | 0.029 |
| `tentTuned` | 81.23 | 82.74 | 80.77 | 76.87 | 1.169 |
| `consistencyTent` | **81.86** | **83.18** | **81.47** | **77.64** | 0.644 |
| `oracle` | 80.17 | 81.81 | 79.68 | 75.33 | 0.036 |

seen 与 unseen 两列上 `consistencyTent` 相对 `tent` 各高 1.44 与 1.89 pp，相对 `tentTuned` 各高 0.44 与 0.70 pp，而它
每格走的距离 0.644 是 `tentTuned` 的 55%。相对错误降低率从 `bnAdapt` 的 15.0% 递增到 `consistencyTent` 的 23.0%。
15 类腐蚀中有 13 类的最高读数属于 `consistencyTent`；`contrast` 归 `tentTuned`，`jpeg_compression` 归 `sourceOnly`。

unseen 按 severity 分档，适应收益随强度增长；最轻的一档上只有 `consistencyTent` 仍在冻结模型之上（+0.32 pp），
severity 5 上达 +12.92 pp。

### 3.5 在线流：排序同向，batch 16 是边界

| batch | `sourceOnly` | `bnAdapt` | `tent` | `tentTuned` | `consistencyTent` | `oracle` |
|---|---|---|---|---|---|---|
| 16 | 67.56 | 65.16 | 66.46 | 29.63 | 53.78 | **68.20** |
| 64 | 67.56 | 67.35 | 67.81 | 67.04 | **69.62** | 68.51 |
| 256 | 67.55 | 67.98 | 68.10 | **71.27** | 70.05 | 68.29 |

门控变体在 batch 64 与 256 上高于同 batch 的 `tent`（+1.81 与 +1.95 pp）；batch 64 还是六个方法里全表最高的一行，
69.62% 高于读标签的 `oracle`；batch 256 的最高读数属于 lr 0.064 的 `tentTuned`。

batch 16 上两个放大更新量的方法塌陷：`consistencyTent` 53.78%（−13.78 pp），`tentTuned` 29.63%（−37.93 pp），同档
冻结模型读 67.56%。整条流走完的参数距离把边界画在位移上：batch 16 上 `tent` 0.294、`tentTuned` 6.490、
`consistencyTent` 2.721、`oracle` 0.405；batch 越小，整条流位移越长，塌陷发生在距离最长的一行。

### 3.6 失效边界与 batch 大小

batch 16 的流上，四个不读标签的方法全部低于冻结模型的 67.56%：`tent` 66.46%（−1.10 pp）、`bnAdapt` 65.16%
（−2.40 pp）、`consistencyTent` 53.78%（−13.78 pp）、`tentTuned` 29.63%（−37.93 pp）；读标签的 `oracle` 68.20%
高于它们。

塌陷的形状在段这一级才看得清：`bnAdapt`、`tent`、`oracle` 最差的段是第一条 `shot_noise`；两个放大更新量的方法最差的
段是第三条 `fog`，`tentTuned` 在那里只剩 20.50%，而同一段上 `tent` 还读 62.60%。batch 64 与 256 没有这种逐段加深。

batch 大小本身还在起作用：同一 lr 0.001 下 TENT 随 batch 下降，64 → 77.90%、32 → 77.30%、16 → 75.45%；给 batch 16
单独铺一条阶梯，六个点单调升到 76.22%，仍未接近 batch 64 那一段；把位移配平之后差距还在（0.889 处 76.22% 对 0.850
处 78.95%，低 2.73 pp）。

放大更新量的两条通道（λ 与步长）在逐格重载的验证格上都是正读数：batch 16、lr 0.001 时 λ=4 比同格同 lr 的 TENT 高
1.34 pp，把步长放大到 0.016 高 0.77 pp。同一批放大放进段间不重置的流、batch 压到 16，读数是 −13.78 与 −37.93 pp。
两处的差异来自 BN 统计量的累积方式：静态格每格回到冻结权重，流上 25,000 张图的仿参更新一路带着走，四倍步数放大的是这条
累积路径。

### 3.7 按样本挑选的上限：65 格上 0.09 pp

`oracle` 读标签，给出"按更新前对错筛样本"这一贪心规则的上限：65 格静态网格上 80.17%，比 `tent` 高 0.09 pp；在线流
三个 batch 上 68.20 / 68.51 / 68.29%，比 `tent` 高 1.74 / 0.70 / 0.19 pp。同一批验证格上把 lr 从 0.001 提到阶梯峰值
0.064 得到 +1.05 pp（不同一批格子，只作量级比较）。

按样本挑选可用的空间就是这个量级：读标签给 +0.09 pp，把每批更新量放大给 +1.05 pp。`consistencyTent` 在 65 格上相对
`tent` 的 +1.78 pp 落在后一条通道里，扣掉位移之后余下的 +0.16 ± 0.13 pp 见 3.3。

### 3.8 方法的开销

三行由 `evaluation/efficiency.py` 的同一个 `measure` 在 batch 64、MPS 上测得：

- 纯前向：5,051 张/秒、12.67 ms、峰值 67.1 MB；
- 一次 TENT 步：2,235 张/秒、28.64 ms、峰值 67.2 MB；
- 一次一致性步：1,138 张/秒、56.25 ms、峰值 66.9 MB。

第二个视图使单步延迟升到 TENT 的 2.0 倍，三次运行的比值落在 1.96–2.00；三行峰值内存都是 67 MB，增广视图不增加内存。
绝对延迟随设备负载变化，比值与内存不随负载变化。

## 4. 项目结构

| 层 | 位置 | 作用 |
|---|---|---|
| 决定 | `configs/` + `configIO.py` | 冻结配置：格子划分、门控形式、事先固定的判定标准 |
| 数据 | `dataset/` | 读 CIFAR-10/10-C、归一化、在线流拼接 |
| 模型 | `models/` | ResNet-18 骨干；只允许 BN scale/shift 更新的接口 |
| 方法 | `methods/` | 六个方法，统一 `predict(batch, targets) -> logits` 契约 |
| 评测 | `evaluation/` | 静态格/流批循环、汇总统计、位移、每批延迟与内存 |
| 运行与报告 | `run/` + `results/` | 单一入口分派九个阶段；无参数时只读 CSV 并重画图 |
| 测试 | `tests/` | 26 个纯函数 CPU 用例，不读数据、不读 checkpoint、不写文件 |

**关键配置：**

- `base.yaml`：公共设置，包括 seed、设备优先序、源模型训练配方、适应阶段参数。
- `splits.yaml`：验证格 10 个、测试格 65 个、在线流 25,000 张的划分。
- `gate.yaml`：门控形式（`entropyGate.use: false`、`consistencyGate.weight: 16.0`）与 `ablation`、`ladder` 两张
  变体表。
- `pilot.yaml`：事先固定的两条判定要求与阈值。

## 5. 复现要点

### 5.1 环境

```bash
cd "AMA564 Project"
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.lock.txt
```

`requirements.lock.txt` 将 torch 固定为 2.10.0、torchvision 固定为 0.25.0；更新的 torch 在 MPS 上会让 ResNet 的全连接
层中止进程。若解释器没有可用的系统信任库，首次下载 CIFAR-10 前需显式指定证书路径：

```bash
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")
```

### 5.2 数据与 checkpoint

```bash
curl -L -C - -o data/raw/CIFAR-10-C.tar \
  https://hf-mirror.com/datasets/Sijiax/CIFAR-10-C/resolve/main/CIFAR-10-C.tar
shasum -a 256 data/raw/CIFAR-10-C.tar   # 应为 c72763e1...cb876a7，2918471680 字节
tar -xf data/raw/CIFAR-10-C.tar -C data/
```

CIFAR-10 由 `dataset/cifar10Source.py` 首次使用时自动下载。来源、数组形状、severity 块校验记在 `data/DATA.md`。

`models/checkpoints/resNet18Seed7.pt` 为 44,805,323 字节，SHA-256
`98392a5091950ea0d758480d89aea1835ed93781b45c5f96af7a604d2a8b15b8`，第 3 节每一行都由它产生。仓库不分发这份权重；新
机器上可用 `run.runAll train` 以 seed 7 和同一配方重训，但 MPS 卷积核不保证逐位一致，重训文件校验和通常不同，核对第 3
节时以校验和为准。

### 5.3 启动命令

全部命令通过同一个入口：

```bash
.venv/bin/python -m run.runAll <命令>
```

| 命令 | 产出 | 重现结论必需 | 作用 |
|---|---|---|---|
| 不带参数或 `report` | 打印 | 是 | 只读 `results/*.csv`：冻结配置、主表、流表、等位移表，并重画五张图；不需要数据与 checkpoint |
| `check` | 打印 | 是 | 上面全部，外加实际运行十个验证格，与 `pilot.csv`、`ladder.csv` 的 40 个精度逐格比对，并打印 `oracle` 等价自检四行 |
| `pilot` | `pilot.csv`(30) | 是 | 三个参照方法依次运行十个验证格，打印判定结果 |
| `gate ablation` / `gate ladder` | `ablation.csv`(420)、`ladder.csv`(560) | 是 | 运行 `gate.yaml` 中命名的变体表 |
| `gradient ladder` | `gradientScale-ladder.csv`(26) | 是 | 测量 26 个变体每批的三个梯度范数 |
| `main [方法名…]` | `mainTable.csv`(1,170) | 是 | 65 个测试格 × 6 方法 × 3 seed |
| `stream sweep` | `stream.csv`(324) | 是 | 在线流上六个方法；仓库里的 324 行覆盖三个 batch，重现结论需带 `sweep` |
| `train` | checkpoint + `sourceTrain.json` | 只在新机器 | 从零训练冻结的 ResNet-18 |
| `throughput` | 打印 | 否 | 测量三类单步的吞吐与峰值内存；需要数据，不需要 checkpoint |
| `figures` | `results/figures/*.png`(5) | 否 | 只重画五张图 |
| `test [模块名]` | 打印 | 否 | 跑 `tests/` 的 26 个用例，CPU 张量，不需要数据与 checkpoint |

每张 CSV 由写它的那条命令独占：运行前先清空目标文件再逐行写入。唯一例外是 `main` 指定方法名时，它只运行子集并保留已有
行。命令名不在表内时，入口打印这张表并以状态码 2 退出。

### 5.4 已验证范围

- 把 `data/` 与 `models/checkpoints/` 同时改名后运行不带参数的命令：冻结配置、主表、流表、等位移表全部照常打印，五张图
  全部重画；第 3 节不需要数据与 checkpoint 就能核对。
- `check`：`oracle` 等价自检四行为 1.803e-04、0.000e+00、0.000e+00、0.000e+00；十个验证格上 40/40 个 live 精度等于
  CSV 行，max |delta| 0.0000 pp。
- `pilot`：重写 30 行，`accuracy` 与 `meanEntropy` 逐行等于重写前文件，只有 `seconds` 变化；判定行仍是
  `FAIL -> switch to plan B`。
- `gradient ladder`：重写 26 行，八个数值列逐行相等；λ=16 在 batch 64 上的 `termRatio` 为 45.01。
- `figures`：两轮重画的五张 PNG 与重画前五份文件 MD5 全部相同。
- `throughput`：MPS 上三次运行对 TENT 延迟比值 1.96–2.00，峰值内存三档都是 67 MB；CPU 分支读整个进程峰值 RSS；屏蔽
  `resource` 导入时内存列打印 `n/a`，命令跑完退出。
- `test`：26 个用例全部通过，退出码 0；`testGateWeights` 打印的两个跨 batch 差值是 0.206 与 0.022。

`main`、`stream`、`gate ablation`、`gate ladder` 四段本轮没有整段重跑；`results/` 里现有的 1,170、324、420、560 行由
当前代码在当前配置下写出。写完后改动过三处，均不触及这些读数：`batchOrder` 补齐修正只在 `count` 小于一个 batch 时改变
输出；`efficiency.py` 只被 `throughput` 使用；`gate.yaml` 本轮只加注释，键值未动。

## 6. 结论与贡献

**有效结论：**

- TENT 相对冻结模型的主要增益来自 BN 统计量重估，熵目标的贡献有限。
- 两道门里真正有效的是**一致性损失项**；它的实际作用点是放大每批更新量，等位移配平后仅余 +0.16 ± 0.13 pp。
- 按样本筛选的空间有限：读标签的 `oracle` 在 65 格上只比 TENT 高 0.09 pp。
- 失效边界可定位：在线流 batch 16 上放大更新量的方法显著低于冻结模型；`jpeg_compression` 的损失来自统计量替换本身，
  门控碰不到。
- 代价可量化：单步延迟为 TENT 的 2.0 倍，峰值内存同量级（约 67 MB）。

**贡献定位：** 本项目的核心产出是一组诊断性结论：筛选机制的实际作用点是步长，步长放大存在 batch 下限，方法有明确的
失效边界和代价。方法在 batch 64/256 与静态测试格上的读数领先，这一领先主要与等效放大步长同向，无法单独归因于
"筛得更准"。

**局限：** 结论限于一份 ResNet-18 checkpoint、CIFAR-10-C 这批格子、三个 seed、batch 16/64/256；未覆盖更大模型、其他
数据域或真实端侧部署。`main`/`stream`/`gate` 四段未整段重跑，已有 CSV 由当前代码在当前配置下写出，后续改动不触及这些
读数。
