# AMA 564 课程项目 —— 一致性门的作用点在步长上：TENT 在混合在线分布漂移下的实证研究

## 结论

1 节记下的预期是：把"值得更新的样本"筛出来，精度应当高于不加筛选的 TENT。以下读数限于这份 checkpoint、这批
格子和三个 seed。逐格回到冻结权重的协议下，带一致性项的方法在 65 格上是六个方法中最高的：`sourceOnly` 76.45%、
`bnAdapt` 79.97%、`tent` 80.08%、`oracle` 80.17%、`tentTuned` 81.23%、`consistencyTent` 81.86%。机制那一侧的
读数是 +1.78 pp 里等位移对照余下 +0.16 ± 0.13 pp，其余部分跟着一致性损失项把每批的更新量放大 22 倍走
（0.644 对 `tent` 的 0.029）。两道门里最终进入目标的是 λ=16 的一致性损失项，熵门关闭，加权角色的读数在噪声内。

- **筛选样本这条设计里成立的是损失项。** 一致性分数作为损失项加入，λ=1 就有 +0.23 pp（10 个验证格全为正），
  λ=16 到 +1.26 pp。熵门单独使用是 −0.01 pp（q=0.75）与 −0.03 pp（q=0.5），叠在 λ=16 上把 +1.26 pp 削成
  +0.84 与 +0.67 pp；加权角色 +0.01 pp，10 格里 5 正 2 负 3 持平。
- **λ=16 是内点。** 验证格上 λ 取 1 / 2 / 4 / 8 / 16 / 32 / 64 得到 +0.19 / +0.33 / +0.59 / +0.91 / +1.26 /
  +1.23 / +0.16 pp，峰值两侧都有实测档位。32 与 64 的均值接近峰值或回落，格间散布同时放大到 −3.43～+3.30 与
  −8.39～+2.63 pp。
- **预注册的 pilot 闸门：未通过。** 均值增益 +10.97 pp 满足第一条要求，`jpeg_compression` 的平均 −2.20 pp
  否定第二条，`run.runAll pilot` 打印 `FAIL -> switch to plan B`。研究按预设分支转向 batch 与混合流的实证比较，
  门控作为被测量的对象保留，最终方法在 65 格主表上排在首位。
- **验证格上选出的超参迁移到陌生腐蚀。** `consistencyTent` 在 seen 15 格、unseen 50 格、unseen severity 5 三组
  分别是 83.18%、81.47%、77.64%，三组里都排第一；`tentTuned` 的同一读数是 82.74%、80.77%、76.87%。
- **失效边界存在，两处。** batch 16 的在线流上，两个放大更新量的方法跌到冻结模型之下（`consistencyTent` 53.78%、
  `tentTuned` 29.63% 对 `sourceOnly` 67.56%），跌的深度沿位移轴加深；`jpeg_compression` 上冻结模型 84.21% 高于
  全部适应方法，损失出在用 batch 统计量替换 running 统计量这一步本身。
- **在线流上的排序随 batch 改变。** batch 64 上 `consistencyTent` 的 69.62% 是六个方法中最高的，batch 256 上
  70.05% 低于 `tentTuned` 的 71.27%，batch 16 上是 53.78%。同一段放大在每格回到冻结权重的静态格上读数为
  +1.34 与 +0.77 pp，在段间不重置的流上读数为 −13.78 与 −37.93 pp，差别在 BN 统计量的累积方式。
- **低成本这一预设：代价是单步延迟翻倍。** 第二个视图使单步延迟升到 TENT 的 2.0 倍，三档测量的峰值内存都是
  67 MB，可更新的只有 9,600 个 BatchNorm 仿参，三行读数见 3.8。
- **按样本挑选可用的空间比预设的小。** 读标签的 `oracle` 在 65 格上带来 +0.09 pp；同一批验证格上把 lr 从
  0.001 提到阶梯峰值 0.064 带来 +1.05 pp。

最终方法在静态网格与在线流的高 batch 档上都排在六个方法之首，它带来的增益沿着一致性损失项放大每批更新量这条
通道走。`configs/gate.yaml` 的两个开关与上面这组读数一致：`entropyGate.use: false`、`consistencyGate.weight: 16.0`，
两个角色都留在目标里。

## 项目结构

### 整体框架

| 层 | 位置 | 作用 |
|---|---|---|
| 决定 | `configs/` + `configIO.py` | 冻结下来的设置：调参与评测用的格子划分、门控的形式、预注册的判定规则 |
| 数据 | `dataset/` | 读 CIFAR-10 与 CIFAR-10-C，归一化，将多种腐蚀按顺序拼接为一条在线序列 |
| 模型 | `models/` | ResNet-18 骨干，以及"只允许动 BatchNorm 的 scale/shift"这个接口 |
| 方法 | `methods/` | 六个方法，全部实现同一个 `predict(batch, targets) -> logits` 契约 |
| 评测 | `evaluation/` | 两种批循环（静态格子、不重置的流）、汇总统计、每批延迟与内存 |
| 运行与报告 | `run/` + `results/` | 一个入口分派九个阶段并把行写进 CSV，不带参数时只读表与图；所有产出都在 `results/` 中 |
| 测试 | `tests/` | 纯函数的单元测试，跑在 CPU 张量上，不需要数据与 checkpoint，由同一个入口的 `test` 命令发现 |

### 逐层细节

#### `configs/` ＋ 根目录的 `configIO.py`

`configIO.loadConfig` 是全仓库唯一的 `yaml.safe_load`，`configIO.setPath` 供 `runAblation.variantGate` 覆写
变体键。每份 YAML 各自冻结一类配置：

- **`base.yaml`** —— 一次实验的公共设置：
  - 运行环境：seed 7、设备优先序 `[mps, cuda, cpu]`、四个目录路径；
  - 源模型训练配方：100 epoch、batch 128、SGD lr 0.1 加 cosine、momentum 0.9、weight decay 0.0005，输入端随机
    裁剪与水平翻转；
  - 适应阶段：`bnMode: train`、静态格逐格重载 `resetPerCell: true`、共享 lr 0.001、TENT 自己的
    `tunedLr 0.064`、每批一步、batch 64、batch 扫描 `[16, 64, 256]`、`seeds [7, 8, 9]`。
- **`splits.yaml`** —— 决定哪批格子用于调参、哪批格子用于出结论：
  - 验证格：10 个（见过的 5 种腐蚀 × severity 3、5，每格 5,000 张），永远不进结论；
  - 测试格：65 个（见过的 15 格取 severity 1/2/4，没见过的 50 格取 1–5，每格 10,000 张）；
  - 在线流：5 种没见过的腐蚀、severity 5、每种 5,000 张，按顺序拼接。
- **`gate.yaml`** —— 定义门控的形式与两张变体表：
  - 最终方法是 `entropyGate.use: false` 加 `consistencyGate: {use: true, weight: 16.0, inLoss: true,
    inWeights: true}`，λ 的七档取值连同各自的精度与位移整个写在 `ablation` 表里；
  - 变体表是 `ablation`（14 个变体 × 3 seed：两个门、两个角色、λ 取 1 到 64 七档）与 `ladder`（26 个变体：
    batch 64 上 TENT 的九个步长与 λ 的五个档位各三 seed，另有 batch 16 的九行与 batch 32 的两行对照）；每张表的
    每一行都记录了它覆写的键。
- **`pilot.yaml`** —— 预注册的判定规则，在任何结果产生之前就已确定；两条阈值与实测判读放在第 1 节。

#### `dataset/`

- **`cifar10Source.py`** —— 训练源模型所需的数据：
  - `loadSplit(root, train)` 取出指定划分的图像与标签（uint8 `[n,32,32,3]`、int64 `[n]`），首次使用时自动
    下载 CIFAR-10；
  - `trainingStatistics` 在训练划分上拟合归一化均值与方差，checkpoint 中存储的就是这两个向量。
- **`cifar10CLoader.py`** —— 从解压后的 npy 文件读取数据：
  - `cell(root, 腐蚀, severity, 张数)` 取出一格；
  - `blockSlice` 按 severity 切块；
  - 用 `mmap_mode="r"` 一次只打开一个腐蚀数组（`(50000,32,32,3)` uint8，五档各 10,000 张）。
- **`preprocess.py`** —— 将 uint8 转为可反传的输入：
  - `normalisationTensors` 将归一化统计量转为设备上的张量；
  - `prepare` 把 uint8 转成 float 并归一化（每格一次），结尾 `.contiguous()`：train-mode BatchNorm 的反向
    不接受 permuted stride 的输入。
- **`onlineStream.py`** —— `buildStream` 将几种腐蚀按顺序拼接成一个序列，返回图像、标签和 `sourceId`，后者让评测
  能按腐蚀分段读数。

#### `models/`

- **`resNet18.py`** —— 骨干与"只动 BN"的接口：
  - `buildResNet18` 构建 torchvision 的 ResNet-18 骨干；
  - `keepBnAffineOnly` 冻结除 BN 的 scale/shift 外的一切，并统计剩余可训练参数量（9,600 / 11,181,642）；
  - `loadSourceModel` 从 checkpoint 重建模型，并返回其中的归一化统计量。

#### `methods/`

- **`losses.py`** —— 两个逐样本的量：
  - `predictiveEntropy` 是 TENT 的目标；
  - `symmetricKl` 是两张视图之间的分歧，即一致性分数 `d_i`。
- **`baselines.py`** —— 两个不做梯度步的参照：
  - `makeSourceOnly` 使用 eval 模式，用源训练时的 running 统计量；
  - `makeBnAdapt` 使用当前 batch 统计量、权重不变，即 AdaBN。
- **`tent.py`** —— TENT 本体与它的上限变体：
  - `makeTentStep` 执行一步熵最小化；
  - `makeTentPredict` 在更新之后于同一批上取 logits；
  - `labelMask=True` 把熵损失限制在"更新前已预测对"的样本上，这就是 `oracle`，它不需要额外一次前向。
- **`gatedTent.py`** —— 门控变体 `consistencyTent` 的实现：
  - `makeView` 用一次 batched `grid_sample` 生成第二视图（每图随机裁剪回 32×32、再按 0.5 概率水平翻转，只使用
    几何变换）；
  - `normaliseDisagreement` 将一批分歧缩放到 0–1；
  - `sampleWeights` 给出 `w_i = max(1 − d_i, 0.2)`，熵门开启时再乘一个 0-1 掩码，权重一律 `detach`；
  - `gatedObjective` 返回 `sum_i w_i (H_i + λ d_i) / sum_i w_i`，连同逐样本的熵与分歧。

  `keepFraction 1.0` 且不启用一致性门时它恰好退化成平均预测熵，也就是 TENT —— 所有"相对 TENT 的差值"都以这条
  退化关系为前提。

#### `evaluation/`

- **`protocols.py`** —— 两个批循环的实现：
  - `cellBatches` / `streamBatches` 生成批；
  - `runCell` / `runStream` 边适应边分类；
  - `scoreBatch` 返回逐图对错与这批平均熵；
  - `cellList` / `testCellList` 从 `splits.yaml` 生成格子列表（见过的在前）；
  - `snapshot` / `displacement` 在每格前后取可更新参数的副本，量出这一格实际走过的距离；
  - `evaluateCells` 是静态格的循环，按 `resetPerCell` 决定逐格重载 checkpoint 还是带着上一个格子的参数继续；
  - `evaluateCell` 是 `runPilot` 用的单格封装。
- **`metrics.py`** —— 表中的汇总量：
  - `relativeErrorReduction` 算相对错误降低率；
  - `meanStd` 给跨 seed 的均值与总体标准差；
  - `grouped` 按格子键的一处分组取均值，0 是腐蚀、1 是 severity；
  - `restrict` 把逐 seed 的表限制在指定的腐蚀上；
  - `cellMeans` 取每个 seed 表在全部格子上的均值；
  - `acrossSeeds` 先按组算每个 seed 的均值，再跨 seed 汇总成均值±标准差。
- **`efficiency.py`** —— `measure(step, batches)` warmup 5 批、最多 50 批，返回每批毫秒、每秒张数与峰值内存；
  前向和两种更新步都由它测量，因此第 3.8 节的 2.0× 比值两侧口径一致。`allocatedMb` 按设备分支：mps 与 cuda 读
  框架自己的分配量，CPU 读整个进程的峰值 RSS（含解释器与 torch，因此与前两者不可比），Windows 没有 `resource`
  模块，该分支返回 NaN，打印成 `n/a`，命令照常跑完。

#### `run/`

- **`runAll.py`** —— 仓库中唯一可执行的模块：
  - 不带参数只读 `results/*.csv`，打印全部表并重画图；
  - `check` 再加上十个验证格的实际运行比对；
  - 其余命令在 `STAGES` 表中逐个指向下面的阶段模块，逐条见 4.3。
- **`runUtils.py`** —— 下面这些部件被所有阶段共用：
  - `resolveDevice` 按 `[mps, cuda, cpu]` 的优先序选取第一个可用的设备；
  - `setSeed` 统一设置 `random`、numpy 与 torch 的种子；
  - `batchOrder` 按 seed 排列一次，再重复序列头部以补齐满 batch；
  - `appendRows` 是全仓库唯一的 CSV 写入处，用 `csv.DictWriter`，首行写表头；
  - `readRows` 用 `csv.DictReader` 读回一张 CSV 的全部行，文件缺失即报错；`makeFigures.readRows` 是宽容版本，
    文件缺失返回 `None`。
- **`trialDriver.py`** —— 一串格子的共用驱动：
  - `buildContext` 组装设备、checkpoint、归一化统计量与输出路径，并把 checkpoint 的模型读成上下文里唯一的一份；
  - `reloadModel` 提供 episodic 协议的每格重置：把冻结权重就地写回那一份模型。为每个格子新建一个 ResNet 会在
    一趟 65 格的测试集上反复分配同一份权重，把进程推进 swap；就地写回与重新加载给出的是同一个状态，`check` 跑的
    就是这条路径，十个验证格上 40 个精度逐格等于 `pilot.csv` 与 `ladder.csv` 里的行；
  - `runTrial` 让一个配置依次运行一串格子并逐行写入 CSV，ablation、阶梯与主表共用它，耗时列的口径见 4.3。

下面的阶段模块各自提供一个 `main(args)`，只被 `runAll.STAGES` 调用，都不能被直接执行：

- `trainSource` 生成 checkpoint；
- `probeThroughput` 输出吞吐数字；
- `runPilot` / `runAblation` / `runGradientScale` / `runMain` / `runStream` 把行写进 `results/`；
- `makeFigures` 生成五张图。

方法名单的扩展关系只有一处：

- `runPilot.methodFactories` 提供三个参照方法；
- `runMain.methodFactories` 在其之上增加 `tentTuned`、`consistencyTent`、`oracle`；
- 打印与作图的顺序由 `makeFigures.METHOD_ORDER` 一处给出。

#### `results/` 与 `data/`

- **CSV** —— 六张表都在仓库里；每张表的写入命令与行数见 4.3 的命令表。
- **`figures/*.png`** —— 五张图，完全由 `makeFigures` 从 CSV 重画；对应的 CSV 缺失时那张图跳过，每写完一张，
  它在终端打印本次读取的行数、方法数与 seed 数：
  - `stepSize`：TENT 的步长阶梯与一致性项的 λ 阶梯置于同一条实测位移轴上，每个 batch 一条曲线，用于把位移本身
    与一致性项的独立效应分开；
  - `ablation`：每个门配置相对 TENT 的均值增益，并标注增益为正的格子数；
  - `severityCurves`：没见过的腐蚀上精度随 severity 的曲线；
  - `perCorruption`：主表按腐蚀类型分组的精度，六个方法并排；
  - `batchSweep`：整条流的精度随 batch 大小的变化。
- **`sourceTrain.json`** —— checkpoint 路径、设备、拟合出的统计量和逐 epoch 曲线。
- **`data/DATA.md`** —— `data/` 整目录写在 `.gitignore` 中，只有这份说明被提交：来源、URL、SHA-256、数组形状、
  severity 块校验、四个腐蚀家族与 seen/unseen 的对应关系。

#### `tests/`

四个模块共 26 个用例，全部跑在 CPU 张量上，不读数据、不读 checkpoint、不写任何文件。`run.runAll test` 用标准库
`unittest` 发现它们，失败时以状态码 1 退出；带一个模块名参数时只跑那一个模块。

- **`testConfigIO.py`**（4）—— `setPath` 返回同一个字典并写进目标键，真实 `gate` 配置的其余键不受影响；未知
  section 抛 `KeyError`，三级路径抛 `ValueError`，后者钉住当前只支持两级键的行为。
- **`testBatchOrder.py`**（7）—— 补齐后的长度是 batch 的整数倍，5,000 → 5,056 与 10,000 → 10,048 就是 2.3 节那
  两个读数；前 `count` 个下标是一个排列，补出来的部分等于序列头部，整除时不补，`count` 小于一个 batch 时补满
  一个 batch；同 seed 相同、异 seed 不同。
- **`testDisplacement.py`**（6）—— 在一个两参数模块上量 `trainableParameters`、`snapshot` 与 `displacement`：
  只列可训练参数，没有可训练参数时位移为 0，冻结参数的移动不计入，[3, 4, 0] 的移动量出 5.0，快照是当时那一刻
  的副本。
- **`testGateWeights.py`**（9）—— 两道门都关时权重全 1，`keepFraction` 1.0 时即便开门也全 1，q=0.5 留下熵最低
  的一半，权重 detached，floor 0.2 是下界，`normaliseDisagreement` 把一批分数铺到 0–1、常数批铺成 0，两个相同
  视图的对称 KL 为 0；最后一个用例在一条固定的 256 维分数向量上量出 2.3 节那两个跨 batch 差值。

## 1. 研究的问题与预想

**问题。**

- 在混合的、在线的分布漂移下，给 TENT 加上"按样本筛选梯度"的目标（熵阈值 + 增广一致性加权）之后的精度变化；
- test-time adaptation 相对冻结源模型失去优势的边界。

**预想。** TENT 对每个样本施加同样的熵最小化，会在已经判错的样本上把误差进一步放大；把"值得更新的样本"筛
出来，精度应当提高。两道门就是这个筛选的两种低成本实现。

**判定规则在实验之前就已确定。** `configs/pilot.yaml` 定下两条要求，同时满足才算通过：

- TENT 在 10 个验证格上平均比源模型高 ≥1.0 pp；
- 任何一种腐蚀的跌幅不超过 0.5 pp。

不满足则转向 batch 大小与混合流的实证对比。实测结果记在 `results/pilot.csv`（十个验证格、seed 7、逐格重载）：
第一条满足（`tent` 相对源模型的均值增益 +10.97 pp），第二条被 `jpeg_compression` 的平均 −2.20 pp 否定，
`run.runAll pilot` 打印的是 `FAIL -> switch to plan B`。研究按预设分支继续：主体是 batch 与混合流上的实证比较，
门控作为被测量的对象保留在结果中。同一张表里 `bnAdapt` 的均值增益是 +10.95 pp，也就是说这一段的绝大部分在只重估
统计量、不执行任何梯度步的方法上同样成立。

## 2. 过程

### 2.1 数据与任务

源模型是 torchvision 的 ResNet-18（11,181,642 个参数），在 CIFAR-10 训练集上从零训练 100 epoch，第 100 epoch
在 CIFAR-10 测试集上 **88.07%**；归一化用的均值与方差也在训练集上拟合。适应阶段能更新的只有其中 40 个 BatchNorm
张量中的 9,600 个仿参，卷积与全连接权重全程冻结。这份权重保存在 `models/checkpoints/resNet18Seed7.pt`，文件名中
的 seed 7 是训练它时用的种子。

88.07% 低于在 CIFAR 上专门调参的 ResNet-18 常见的 ~92%，因为 torchvision 的变体保留了 ImageNet 的
7×7 stride-2 stem 加 maxpool；这个差距来自架构本身，六个方法读的又是同一份权重，比较不受影响。存档的是第 100
epoch 的权重，没有按测试精度回选某一代。

评测面在 CIFAR-10-C 上定义如下：

- 测试格：65 个（见过的 5 种腐蚀 × severity 1/2/4，加没见过的 10 种腐蚀 × severity 1–5，每格 10,000 张）；
- 在线流：5 种没见过的腐蚀在 severity 5 上按顺序拼接，段间不重置；
- batch 档：16 / 64 / 256。

调参只用 10 个验证格，它们永远不进结论。

### 2.2 六个方法

| 方法 | 更新目标 | 读标签 |
|---|---|---|
| `sourceOnly` | 不更新，BN 用源训练的 running 统计量 | 否 |
| `bnAdapt` | 不更新权重，只让 BN 使用当前 batch 的统计量（AdaBN） | 否 |
| `tent` | 平均预测熵，lr 0.001 | 否 |
| `tentTuned` | 同一目标，lr 0.064（验证格选出） | 否 |
| `consistencyTent` | `sum_i w_i (H_i + 16·d_i) / sum_i w_i`，λ=16（验证格选出） | 否 |
| `oracle` | 平均预测熵，只计入更新前已预测正确的样本 | 是，作为上限参照 |

### 2.3 受控变量

- **静态格逐格重载。** `adaptation.resetPerCell: true` 让 `runPilot` 和 `trialDriver.runTrial`（ablation／
  阶梯／主表）在每一格之前重新加载那份冻结 checkpoint，于是一格的读数只取决于这一格；置为 false 时一个模型走完
  整个格子列表，靠后的格子读在前面格子改写过的 BN 统计量上。逐格重载是 CIFAR-10-C 通常的读法，也是全部结论的
  口径。两种口径在同一批验证格上的差实测为：`tent` 链式 78.56%、逐格重载 78.09%，`sourceOnly` 与 `bnAdapt`
  则两种口径相同（它们不执行梯度步）。链式多出的 0.47 pp 来自格子在列表中的位置，而 `tent` 相对 `bnAdapt` 的差
  由 +0.50 pp 缩到 +0.03 pp —— 链式口径把跨格的统计量惯性算进了熵目标的功劳。这条对照只要把开关改回 false 再
  运行一次 `run.runAll pilot` 就能重做。在线流不走这条路：段间不重置是它的定义。
- **同一个冻结 checkpoint，只动 9,600 个 BN 仿参。** 卷积权重全程 `requires_grad=False`，六个方法之间只有
  目标函数不同。三个 seed 只改批顺序与增广采样，源模型始终是这一份；不假设 GPU kernel 层面的 run-to-run 确定，
  所以跨 seed 的数字都以均值±标准差给出。
- **train-mode BN。** `bnMode: train` 表示用当前 batch 的均值方差归一化，六个方法一致；只有 `sourceOnly`
  使用 eval 模式，它的对照作用是统计量不重新校准。
- **共享的批顺序与满 batch。** `batchOrder` 按 seed 排一次，所有方法读同一序列；最后一个 batch 通过重复序列
  头部补齐，因此：
  - 验证格 5,000 张按 5,056 张打分（56 张被计入两次）；
  - 测试格 10,000 张按 10,048 张打分（48 张被计入两次）。

  每行 CSV 的 `images` 列记录的就是实际打分的张数。因此方法间的配对差值不受末批长度差异影响。
- **门控按 batch 归一，权重跨 batch 不可比。** 熵门取本批熵的分位数，一致性权重把本批的对称 KL 除以本批极差，
  同一张图落在不同 batch 里因此拿到不同权重。`tests/testGateWeights.py` 在一条固定的 256 维分数向量上实测：切成
  16 的批，权重相对 256 的批最多差 0.206；切成 64 的批最多差 0.022，权重值域是 [0.2, 1.0]；
  `run.runAll test testGateWeights.py` 重跑这两个读数。损失项 λ·d_i 用逐样本的原始对称 KL，不经过这个归一化。
  冻结配置关掉熵门，`ladder` 表每一行的权重都是全 1（TENT 行关一致性门，loss 行关 `inWeights`），主表与
  `ablation` 只跑 batch 64，这些对照都不跨这条归一化；流表三档 batch 跑完整方法，跨档差值含这一项，而权重角色
  单独在验证格上实测 +0.01 pp（`consistencyWeightOnly` 对 `tent`，见 3.2）。
- **每批一步、posterior 读出。** 执行一步梯度更新之后再用更新后的仿参给同一批分类，这是 TENT 原始代码的口径，
  代价是每个 batch 多一次前向，六个方法在这点上相同。
- **`bnAdapt` 分离统计量重估与熵更新。** 只重估统计量、不执行任何梯度步。`tent` 相对 `sourceOnly` 的增益
  若没同时相对 `bnAdapt` 出现，它属于统计量重估；`runPilot` 将 tent−src 和 tent−bn 两列并排打印。
- **位移逐行实测。** 每一条精度行都自带一个 `displacementNorm`：这一格的 9,600 个 BN 仿参在更新前后的欧氏
  距离，由 `evaluation/protocols.snapshot` 与 `displacement` 在批循环里量出。它替代的是 `lr × ‖∇L‖ × 每格批数`
  那个估计值 —— 估计值在源参数处读梯度，因而漏掉了梯度随熵下降而自行缩小的那一段，实测的位移把这段反馈算了进去。
  三条轴（TENT 的步长、λ、batch）都写在这同一个刻度上，`makeFigures.tentCurveAt` 于是在 TENT 自己测出的阶梯上做
  对数插值，取同位移、同 batch 处的 TENT 读数；两者之差就是扣除位移之后剩下的东西。`runGradientScale` 只留下面
  这半个作用：在源参数处分开量出 ‖∇熵‖ 与 ‖λ·∇一致性‖，说明 λ 为什么表现为一个步长倍率，它不再提供那条轴。
- **`oracle` 作为参照点。** 它给出"按更新前的对错筛样本"这一条贪心规则的读数上限，由 `run.runAll check` 开头
  打印的四行固定，四行都在一批真实图像上量出：全对掩码与 plain TENT 的参数更新差为 0.000e+00，两者的 running
  统计量差也是 0.000e+00（它们看的是同一批），全错掩码下 9,600 个仿参一个没动，而 plain TENT 相对自己的起点移动
  1.8e-04。它读标签，因此不是一个可用的方法，也不是所有筛选规则的上界。
- **seen / unseen 分列。** 15 格与 50 格各自成组，以免在见过的腐蚀上选定的超参被当成泛化能力。
- **调参预算对等。** Guidance 要求 comparable tuning budgets，两侧都在同一批十个验证格上以完整 pass 计数，
  并且都跑到曲线掉头为止，因此 `adaptation.tunedLr` 与 `consistencyGate.weight` 都是两侧有实测点的内点：
  - 门：`configs/gate.yaml` 的 `ablation` 表，14 个变体 × 3 seed = 42 次 pass。熵门取 `keepFraction`
    0.5 / 0.75 / 1.0 三档，一致性项的两个角色各能单独开关，λ 取 1 / 2 / 4 / 8 / 16 / 32 / 64 七档；每行的
    `overrides` 列写明它覆写了哪些键。
  - TENT：`ladder` 表中 batch 64 的九个步长 0.0005–0.128 × 3 seed = 27 次 pass，`adaptation.tunedLr` 就读自
    这条阶梯；同一张表里的 λ 阶梯与它共用一条实测位移轴。
  - 阶梯另带 batch 32 的一对照与 batch 16 的九行，各 1 seed，读作那两个区域的形状，不读到两位小数。

  主表因此同时有共享 lr 的 `tent` 和被选出的 `tentTuned`。

### 2.4 判定指标

主指标是每格的分类精度，聚合口径如下：

- 65 格均值；
- seen / unseen 分组均值；
- 按 severity 的分组均值。

以上每个数都给三个 seed 的均值±标准差；同一 seed 读同一批序列，差值才是配对的。辅助指标：

- **相对错误降低率** `(方法 − 源) / (1 − 源)`，以源模型未纠正的错误为分母；
- **同位移处的精度差**，在实测的 `displacementNorm` 轴上对着 TENT 自己的阶梯打分，按 seed 配对后给出均值与
  散布；
- **最差格与最差段**，用来定位适应失效的起点。

预测熵用作优化目标，同时作为置信度实际变化的诊断，不作为评判标准：每张精度 CSV 都把 `accuracy` 与
`meanEntropy` 并排记录，但两者不成比例。contrast severity 3 这一格上三个 seed 的均值（`results/ablation.csv`）
给出两个方向的读数：`tent` 81.43%、批次平均熵 0.2182；λ=16 读到 82.14%、熵 0.2105；λ=32 把熵压到 0.2055、精度
82.25%；λ=64 的熵与 λ=32 只差 0.0001（0.2056），精度却跌回 81.20%。熵门单独保留 75% 时熵升到 0.2200，精度仍是
81.43%。熵的高低排不出精度的顺序，λ=16 的增益因此不能改写成"置信度提高了"，主指标取精度，熵承担诊断。
此外报告每批延迟、峰值内存和被更新的参数量。

## 3. 结果与分析

### 3.1 增益分成两个成分：统计量重估与熵目标

10 个验证格（`results/pilot.csv`，seed 7、batch 64、逐格重载）：

- `sourceOnly` 67.12%；
- `bnAdapt` 78.06%；
- `tent` 78.09%。

从 67% 到 78% 的这一段全部来自 BN 改用当前 batch 的统计量，熵目标在 `bnAdapt` 之上增加 +0.03 pp：十个格子里
六个为正，逐格的配对差值落在 −0.02 到 +0.08。65 格测试表给出同一对成分的另一组读数，`sourceOnly` 76.45%、
`bnAdapt` 79.97%、`tent` 80.08%，也就是统计量重估 +3.52 pp、熵目标再 +0.11 pp；分组看是 seen 15 格 +0.09 pp、
unseen 50 格 +0.11 pp。两个格子集合给出的比例一致：lr 0.001 的那一步几乎全部叠在统计量重估之上。

15 类腐蚀中冻结模型高于全部适应方法的只有 `jpeg_compression`（三档 severity 的均值：`sourceOnly` 84.21%、
`bnAdapt` 81.53%、`tent` 81.60%、`consistencyTent` 83.14%）。损失出在统计量那一侧：severity 1/2/4 上
`bnAdapt` 分别比 `sourceOnly` 低 2.70、2.73、2.60 pp，`tent` 与它的差只有 0.02 到 0.08 pp。一致性项把回升
放大到 +1.54 pp，仍是这类腐蚀上最接近冻结模型的读数。预注册判定里否定第二条要求的 −2.20 pp，读的就是验证格
两档 severity 的 −2.43 与 −1.96。两道门缩放的只有 γ/β 的梯度权重，用 batch 统计量替换 running 统计量由
`bnMode: train` 决定，所以这一类的损失是方法的结构性边界。

### 3.2 增益全部来自一致性损失项，峰值在 λ=16

同一批 10 个验证格、同一 lr 0.001、batch 64、三个 seed，`results/ablation.csv` 的 420 行给出下表；每行的
`overrides` 列记录了它覆写的配置键，"一格位移"是同一行自己记录的 `displacementNorm` 对 30 次运行取均值。
基线 `tent` 为 77.90%。

| 变体 | 配置 | 均值 | 相对 TENT | 一格位移 |
|---|---|---|---|---|
| `tent` | 只最小化熵 | 77.90% | — | 0.018 |
| `entropyGateOnly_q75` | 熵门保留 75%，无一致性项 | 77.89% | −0.01 pp | 0.011 |
| `entropyGateOnly_q50` | 熵门保留 50%，无一致性项 | 77.86% | −0.03 pp | 0.003 |
| `consistencyWeightOnly` | 一致性分数只加权样本（λ=1） | 77.91% | +0.01 pp | 0.017 |
| `consistencyLossOnly` | 一致性分数只作损失项（λ=1） | 78.13% | +0.23 pp | 0.058 |
| `bothGates` | 两个角色都启用（λ=1） | 78.09% | +0.19 pp | 0.045 |
| `bothGates_lambda2` | 两个角色都启用（λ=2） | 78.23% | +0.33 pp | 0.080 |
| `bothGates_lambda4` | 两个角色都启用（λ=4） | 78.49% | +0.59 pp | 0.146 |
| `bothGates_lambda8` | 两个角色都启用（λ=8） | 78.81% | +0.91 pp | 0.267 |
| `bothGates_lambda16` | 两个角色都启用（λ=16，冻结值） | **79.16%** | **+1.26 pp** | 0.473 |
| `bothGates_lambda32` | 两个角色都启用（λ=32） | 79.13% | +1.23 pp | 0.807 |
| `bothGates_lambda64` | 两个角色都启用（λ=64） | 78.06% | +0.16 pp | 1.314 |
| `entropyGate_q75` | 熵门保留 75%，两个角色（λ=16） | 78.74% | +0.84 pp | 0.393 |
| `entropyGate_q50` | 熵门保留 50%，两个角色（λ=16） | 78.56% | +0.67 pp | 0.356 |

七行 `bothGates` 把 `entropyGate.use` 置为 true，`keepFraction` 取文件默认的 1.0 —— `sampleWeights` 的分位截断在
q=1.0 时留下整批、掩码全为 1，所以这七行对应的就是最终方法的形式：一致性的两个角色都启用，λ 依次取 1 到 64。两行
`entropyGate_q*` 只覆写 `keepFraction`，λ 继承文件的 16.0，它们与冻结方法之间只差熵门是否截断这一件事。

四条读数按 10 个验证格的配对差（seed 均值）统计：

- 熵门单独使用为负：q=0.75 是 −0.01 pp（10 格中 3 格为正），q=0.5 是 −0.03 pp（0 格为正）。分位截断去掉的正是
  梯度信号最强的那部分样本，而每批只执行一步、更新量有限，少一批样本的代价大于收益，所以文件把 `entropyGate.use`
  留在 false。
- 同一道门叠在 λ=16 上把 +1.26 pp 削成 +0.84 pp 与 +0.67 pp，一格位移同时从 0.473 缩到 0.393 与 0.356。两行
  的 10 格配对差仍全为正，说明它们付出的是样本量，方向信息没有改变。
- 加权角色的 +0.01 pp 在噪声之内：10 个格的配对差为 5 正、2 负、3 持平，30 个逐 seed 逐格读数为 11 正、8 负、
  11 持平。按分歧大小重排样本权重，在这个设定下没有可检出的作用。
- 可检出的增益都在损失项：λ=1 单独作损失项即 +0.23 pp（10 格全为正），随 λ 升到 16 取到 +1.26 pp，32 停在
  +1.23 pp，64 回落到 +0.16 pp。峰值两侧都有实测读数，`consistencyGate.weight: 16.0` 因此是一个内点。32 的均值
  与 16 相差 0.03 pp，格间散布却放大到 −3.43 到 +3.30 pp，64 更是回落到 +0.16 pp（散布 −8.39 到 +2.63 pp），
  回落并不平滑。3.3 节的位移阶梯与 3.4 节的主表都在这一个 λ 上运行。

### 3.3 位移解释了增益的主体，等位移处余下 +0.16 pp

`results/ladder.csv` 把两条阶梯放在同一批验证格、batch 64、三个 seed 上，每行自己记录一格走过的距离。按实测位移
升序，两条阶梯是同一把尺度上的两组点（精度为三 seed 均值）：

- TENT 沿步长，lr 取 0.001 的 0.5 / 1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 倍：每格位移 0.009 → 1.517，精度
  77.88 / 77.90 / 77.95 / 78.05 / 78.27 / 78.54 / 78.87 / **78.95** / 77.68；
- 一致性项只作损失项，λ 取 1 / 4 / 8 / 16 / 32：每格位移 0.058 → 1.075，精度 78.13 / 78.62 / 78.97 /
  **79.15** / 78.53。

两条阶梯都在位移越过 0.64–0.85 之后回落。最高点分别是 lr 0.064 的 78.95% 与 λ=16 的 79.15%，后者用更短的一格
（0.636 对 0.850）取到更高的精度。

位移取实测值，原因在两处。其一是反馈：九个 TENT 点上实测位移对 lr 的幂指数是 0.92，即步长放大 64 倍只换来
47 倍的距离；`runGradientScale` 在源参数处量到 λ=16 的一致性梯度是熵梯度的 45 倍，这条阶梯真正走过的却是 35 倍
（0.636 对 lr 0.001 的 0.018）。其二是估计值随位移漂移：`lr × ‖∇L‖ × 每格批数` 与实测的比值从 5.9（最小的点）
升到 9.0（最大的点），轴的刻度因此不成比例。不过这两个口径对等位移对照给出的读数几乎相同：把六个 λ 档位分别放到
两条轴上插值，五个档位的参考点差在 0.02 pp 内，λ=32 差 0.11 pp；用实测精度配估计轴，六个差值仍然全为正，范围
+0.12 到 +0.27 pp。换轴没有改变这组符号，改变符号的是 2.3 节那个协议开关。

`makeFigures.matchedGaps` 用每个 seed 自己那条 TENT 阶梯插值，所以同一 seed 的六个 λ 档位是配对读数。下表是
`run.runAll report` 打印的 batch 64 六行，"TENT 同位移处"一列等于精度减去差值：

| λ | 每格位移 | 精度 (%) | TENT 同位移处 (%) | 差 (pp) | seed 间散布 |
|---|---|---|---|---|---|
| 1 | 0.058 | 78.13 | 78.03 | +0.10 | 0.03 |
| 4，lr 0.0005 | 0.105 | 78.29 | 78.19 | +0.10 | 0.06 |
| 4 | 0.199 | 78.62 | 78.42 | +0.20 | 0.11 |
| 8 | 0.363 | 78.97 | 78.73 | +0.24 | 0.12 |
| 16 | 0.636 | 79.15 | 78.91 | +0.24 | 0.18 |
| 32 | 1.075 | 78.53 | 78.43 | +0.10 | 0.15 |

18 个读数的均值 +0.16 ± 0.13 pp，其中 16 个为正；按 seed 汇总是 +0.04 / +0.28 / +0.16 pp（seed 7 / 8 / 9）。
batch 16 的三个单 seed 档位给出 +0.19 / +0.26 / +0.60 pp。batch 32 只有一个 TENT 点，那条轴上无法插值，所以
两个 batch 32 行没有差值。

六个档位在三个 seed 上的均值差全为正。这节可检出的结论因此是两层的：一致性项相对 lr 0.001 的 TENT 高出 1.25 pp
（λ=16 的 79.15% 对 77.90%），其中约五分之一（+0.24 pp）出现在位移相同处，其余部分与单纯把步长放大所得的相同。图见 `results/figures/stepSize.png`。

### 3.4 65 格上三个分组的排序一致

65 格主表（`results/mainTable.csv`，3 seed、batch 64、逐格重载，粗体标每列最高）：

| 方法 | 65 格均值 | seen 15 格 | unseen 50 格 | unseen severity 5 | 一格位移 |
|---|---|---|---|---|---|
| `sourceOnly` | 76.45% | 78.10% | 75.95% | 64.72% | 0.000 |
| `bnAdapt` | 79.97% | 81.65% | 79.47% | 75.03% | 0.000 |
| `tent` | 80.08% | 81.74% | 79.58% | 75.18% | 0.029 |
| `tentTuned` | 81.23% | 82.74% | 80.77% | 76.87% | 1.169 |
| `consistencyTent` | **81.86%** | **83.18%** | **81.47%** | **77.64%** | 0.644 |
| `oracle` | 80.17% | 81.81% | 79.68% | 75.33% | 0.036 |

`consistencyTent` 在四个口径上都是最高的一个，seen 与 unseen 之间同向：相对 `tent` 各高 1.44 与 1.89 pp，相对
被调过的 `tentTuned` 各高 0.44 与 0.70 pp，而它每格走的距离是 0.644，为 `tentTuned` 的 55%。跨 seed 的散布
分别是 0.06 与 0.16，两者都大于 `tent` 的 0.01。相对错误降低率（以源模型未纠正的错误为分母）依次是 `bnAdapt`
15.0%、`tent` 15.4%、`oracle` 15.8%、`tentTuned` 20.3%、`consistencyTent` 23.0%。15 类腐蚀里有 13 类的最高读数
属于 `consistencyTent`，`contrast` 归 `tentTuned`（83.30% 对 83.01%），`jpeg_compression` 归 `sourceOnly`（3.1）。

没见过的 10 类腐蚀按 severity 分档，适应的收益随强度增长，而最轻的一档上只有 λ=16 的这一行还在冻结模型之上
（括号内是相对 `sourceOnly`）：

| severity | `sourceOnly` | `bnAdapt` | `tent` | `tentTuned` | `consistencyTent` |
|---|---|---|---|---|---|
| 1 | 83.79% | 82.54%（−1.25） | 82.62%（−1.17） | 83.42%（−0.37） | **84.11%**（+0.32） |
| 3 | 78.13% | 80.44%（+2.31） | 80.55%（+2.42） | 81.69%（+3.56） | **82.29%**（+4.16） |
| 5 | 64.72% | 75.03%（+10.31） | 75.18%（+10.46） | 76.87%（+12.15） | **77.64%**（+12.92） |

同一张表里走得更远的 `tentTuned` 读数低于 `consistencyTent`，这两行之间的差无法由位移解释。把它算成一致性项
自己的那一份，需要在验证格那条有九个点的 TENT 阶梯上取同位移处，3.3 的配平结果是 +0.16 ± 0.13 pp；测试格与
验证格每格的图像数不同（10,000 与 5,000），两批距离不在同一把尺上，所以这里只作排序的核对。

### 3.5 在线流上的排序与静态网格同向

`results/stream.csv`（5 种未见腐蚀、severity 5、段间不重置、3 seed）：

| batch | `sourceOnly` | `bnAdapt` | `tent` | `tentTuned` | `consistencyTent` | `oracle` |
|---|---|---|---|---|---|---|
| 16 | 67.56% | 65.16% | 66.46% | 29.63% | 53.78% | **68.20%** |
| 64 | 67.56% | 67.35% | 67.81% | 67.04% | **69.62%** | 68.51% |
| 256 | 67.55% | 67.98% | 68.10% | **71.27%** | 70.05% | 68.29% |

粗体标每档 batch 的最高值。门控变体在 batch 64 与 256 上高于同 batch 的 `tent`（+1.81 与 +1.95 pp），batch 64
还是六个方法里全表最高的一行，69.62% 高于读标签的 `oracle`；batch 256 的最高读数属于 lr 0.064 的 `tentTuned`，
71.27% 对 70.05%。batch 16 上两个放大更新量的方法都塌陷，`tentTuned` 29.63%、`consistencyTent` 53.78%，同
一档的冻结模型读 67.56%，`oracle` 68.20%。

于是静态网格与在线流两份读数的方向一致：`consistencyTent` 在 65 格的四个口径、流的 batch 64 与 256 上都是
领先的一行，失效只出现在 batch 16。三条 batch 上整条流走完的参数距离（`displacementNorm` 的 `all` 行，3 seed
均值）把这条边界画在位移上：

| batch | `tent` | `tentTuned` | `consistencyTent` | `oracle` |
|---|---|---|---|---|
| 16 | 0.294 | 6.490 | 2.721 | 0.405 |
| 64 | 0.082 | 2.423 | 1.331 | 0.128 |
| 256 | 0.021 | 0.895 | 0.543 | 0.035 |

塌陷发生在距离最长的一行上，并随 batch 减小而加深：batch 16 每格的更新步数是 batch 64 的四倍，同一行在这一档
走过的整条流距离也长 2.0 到 3.6 倍。lr 0.001 的 `tent` 与 `oracle` 走过 0.294 与 0.405，读数仍在冻结模型的
1.2 pp 之内。Proposal 第 2 页第 5 节预注册的正是这条风险，实测把它定位在 batch 16，3.6 逐段量它。

### 3.6 失效边界在 batch 16 的流上

`results/stream.csv` 的 batch 16 一行中，四个不读标签的方法全部低于冻结模型的 67.56%：

- `tent` 66.46%（−1.10 pp）；
- `bnAdapt` 65.16%（−2.40 pp）；
- `consistencyTent` 53.78%（−13.78 pp）；
- `tentTuned` 29.63%（−37.93 pp）。

读标签的 `oracle` 68.20% 高于它。塌陷的形状在段这一级才看得清：`bnAdapt`、`tent`、`oracle` 最差的段是第一条
`shot_noise`（59.38%、60.44%、62.58%），两个放大更新量的方法最差的段是第三条 `fog`（`tentTuned` 20.50%、
`consistencyTent` 39.08%），同一段上冻结模型读 54.32%、`tent` 读 62.60%。batch 64 与 256 没有这种逐段加深，
两档的最差段读数落在 58.5–65.1% 之间，都高于冻结模型那 54.3% 的一条。

batch 大小本身还在起作用，而这一段无法由位移解释（`ladder` 表的 batch 16 与 32 两段是 seed 7 单跑，batch 64
段取三 seed 均值）：

- 同一 lr 0.001 下 TENT 随 batch 下降：64 → 77.90%、32 → 77.30%、16 → 75.45%；
- 给 batch 16 单独铺一条 0.0005 到 0.016 的阶梯，六个点单调升到 76.22%，仍未接近 batch 64 那一段的读数；把位移
  配平之后差距还在：batch 16 在 0.889 处是 76.22%，batch 64 在 0.850 处是 78.95%，低 2.73 pp；
- batch 32 只有一个 TENT 点，那条轴上配不平，只能给出同一 lr 下的 77.30%。

放大更新量的两条通道在逐格重载的验证格上都是正读数：batch 16、lr 0.001 时 λ=4（只作损失项）比同格同 lr 的
TENT 高 1.34 pp，把步长放大到 0.016 高 0.77 pp（76.22% 对 75.45%）。同一批放大放进段间不重置的流、batch 压到
16，读数是 −13.78 与 −37.93 pp。两处的分别是 BN 统计量的累积方式：静态格每格回到冻结权重，流上 25,000 张图的
仿参更新一路带着走，四倍步数放大的是这条累积路径。`ladder` 里所有 λ 行都把 `consistencyGate.inWeights` 覆写为
false，进入更新的只有损失项，所以"加权把有效 batch 缩小"这条解释在两种场合下都用不上。

### 3.7 按样本挑选的上限：65 格上 0.09 pp

`oracle` 读标签，它给出的上限分两种场合：

- 65 格静态网格：80.17%，比 `tent` 高 0.09 pp（分组看是 seen +0.07、unseen +0.10、unseen severity 5 +0.15）；
- 在线流的三个 batch：68.20 / 68.51 / 68.29%，比 `tent` 高 1.74 / 0.70 / 0.19 pp。

同一批验证格上把 lr 从 0.001 提到 0.016 得到 +0.64 pp，提到阶梯峰值 0.064 得到 +1.05 pp（`ladder` 表，3 seed，
与上面的 65 格不同一批格子，只作量级比较）。按样本挑选可用的空间就是这个量级：读标签给 +0.09 pp，把每批的更新
量放大给 +1.05 pp。`consistencyTent` 在 65 格上相对 `tent` 的 +1.78 pp 落在后一条通道里，扣掉位移之后余下的
+0.16 ± 0.13 pp 见 3.3。

### 3.8 方法的开销

多一个视图就是每批多一次前向。三行都由 `evaluation/efficiency.py` 的同一个 `measure` 在 batch 64、MPS 上测得，
下面是 `run.runAll throughput` 一次运行的打印：

- 纯前向：5,051 张/秒、12.67 ms、峰值 67.1 MB；
- 一次 TENT 步：2,235 张/秒、28.64 ms、峰值 67.2 MB；
- 一次一致性步：1,138 张/秒、56.25 ms、峰值 66.9 MB。

第二个视图使单步延迟升到 TENT 的 2.0 倍，三次运行的比值落在 1.96–2.00；三行的峰值内存都是 67 MB，增广视图不增加
内存。绝对延迟随设备负载变化，比值与内存不随负载变化，所以这条代价以倍数记账，`run.runAll throughput` 在任何
设备上都能重新测得这三行。

## 4. 项目启动

### 4.1 环境

```bash
cd "AMA564 Project"
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.lock.txt
```

`requirements.lock.txt` 将 torch 固定为 2.10.0、torchvision 固定为 0.25.0：更新的 torch 在 MPS 上会让 ResNet
的全连接层中止进程（abort），`pyproject.toml` 中记录了原因。若这个 Python 解释器没有可用的系统信任库，第一次下载
CIFAR-10 前需要显式指定证书路径（`certifi` 已在依赖中）：
`export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")`。

### 4.2 数据与 checkpoint

```bash
curl -L -C - -o data/raw/CIFAR-10-C.tar \
  https://hf-mirror.com/datasets/Sijiax/CIFAR-10-C/resolve/main/CIFAR-10-C.tar
shasum -a 256 data/raw/CIFAR-10-C.tar   # 应为 c72763e1...cb876a7，2918471680 字节
tar -xf data/raw/CIFAR-10-C.tar -C data/
```

CIFAR-10 由 `dataset/cifar10Source.py` 在首次使用时自动下载。来源、数组形状、severity 块的校验记在
`data/DATA.md`。`data/` 和 `models/checkpoints/` 都写在 `.gitignore` 中，因此新克隆的仓库包含代码、配置与结果表；
运行前需要在本地准备：

- 解压得到的 CIFAR-10-C；
- `run.runAll train` 产出的一份 checkpoint。

缺少 checkpoint 时，所有需要它的命令抛出 `FileNotFoundError` 并指明路径 `models/checkpoints/resNet18Seed7.pt`。

仓库不分发这份权重。`models/checkpoints/resNet18Seed7.pt` 为 44,805,323 字节，SHA-256
`98392a5091950ea0d758480d89aea1835ed93781b45c5f96af7a604d2a8b15b8`，第 3 节的每一行都由它产生。
`results/sourceTrain.json` 记下训练代价：100 个 epoch 在 MPS 上合计 3,941.1 秒（65.7 分钟），干净 CIFAR-10
测试集精度 88.07%，归一化统计量与训练配方存在权重文件内。新机器上 `run.runAll train` 以 seed 7 与同一份配方
重训，`setSeed` 固定 random、numpy 与 torch 的 RNG，MPS 卷积核不保证逐位一致，重训文件的校验和因此通常与上面
不同；核对第 3 节时以校验和为准。

### 4.3 启动命令

全部命令都通过同一个入口 `.venv/bin/python -m run.runAll`；命令名不在下表内时，它打印下表并以状态码 2 退出。
下表的阶段模块都在「项目结构」的 `run/` 一节中列出，只由 `runAll.STAGES` 调用，不能被直接执行；`test` 没有阶段
模块，它由 `runAll.runTests` 发现 `tests/`。

| 命令 | 产出 | 对重现结论的必要性 | 作用 |
|---|---|---|---|
| 不带参数或 `report` | 打印 | 是 | 只读 `results/*.csv`：冻结配置、主表、流表、等位移表，并重画五张图；不需要数据与 checkpoint |
| `check` | 打印 | 是 | 上面全部，外加实际运行十个验证格，与 `pilot.csv`、`ladder.csv` 的 40 个精度逐格比对，并打印 `oracle` 的等价自检四行 |
| `pilot` | `pilot.csv`(30) | 是 | 三个参照方法依次运行十个验证格，打印 `pilot.yaml` 的判定 |
| `gate <表名>` | `ablation`(420)、`ladder`(560) | 是 | 运行 `gate.yaml` 中命名的变体表，省略表名时默认 `ablation` |
| `gradient <表名>` | `gradientScale-ladder`(26) | 是 | 测量 `ladder` 表 26 个变体每批的三个梯度范数，等位移那条轴依赖这些测量 |
| `main [方法名…]` | `mainTable.csv`(1,170) | 是 | 六十五个测试格 × 六方法 × 三 seed，打印均值表、逐腐蚀矩阵、未见腐蚀的 severity 矩阵 |
| `stream [sweep]` | `stream.csv`(324) | 是 | 在线流上六个方法；仓库里的 324 行覆盖三个 batch，因此重现结论需要带 `sweep`，省略它只写 batch 64 的 108 行 |
| `train [--epochs N]` | checkpoint + `sourceTrain.json` | 只在新机器上 | 从零训练那份被冻结的 ResNet-18 |
| `throughput` | 打印 | 否 | 测量三类单步的吞吐与峰值内存，也就是 3.8 那三行；需要数据，不需要 checkpoint |
| `figures` | `figures/*.png`(5) | 否 | 只重画五张图，不带参数的运行已经包含它 |
| `test [模块名]` | 打印 | 否 | 用标准库 `unittest` 跑 `tests/` 的 26 个用例，CPU 张量，不需要数据与 checkpoint，不写文件；带模块名时只跑那一个，失败以状态码 1 退出 |

每张 CSV 由写它的那条命令独占：运行前先把目标文件清空再逐行写入，因此中断不会留下不完整的文件；唯一的例外是
`main` 指定方法名的那条命令，它只运行子集并保留已有的行。

字段随命令而变。五张精度表每行都带 seed、方法或变体名、`images`、`accuracy`、`meanEntropy`、`displacementNorm`
与 `seconds`；`pilot`、`ablation`、`ladder`、`mainTable` 另带腐蚀与 severity，`stream` 另带 `batch` 与段名，两张
变体表再带 `overrides`（写明被覆写的键）、`lr` 与 `batch`。梯度表是另一套字段：`block`、`variant`、`lr`、
`batchSize`、`lambdaWeight`、`batchesPerCell`、两个项各自与合计的梯度范数，外加两者之比 `termRatio`。

`displacementNorm` 是这一格把 9,600 个仿参移动的距离，不执行梯度步的方法读 0；流表的五个段行留空，只有 `all`
行带整条流的路径长度。

`seconds` 分两种口径。`pilot.csv` 记一格的时间。`ablation`、`ladder`、`mainTable` 由 `trialDriver.runTrial` 写出，
记一个配置跑完整串格子的墙钟时间，并把这个值重复在该趟的每一行上；`stream.csv` 记一个方法在一个 batch、一个
seed 上跑完整条流的时间，同样重复在该趟的每一行上。整趟耗时因此要先按趟去重再相加，逐行求和会把同一趟重复计入。

λ 与 `keepFraction` 的候选值都留在可以重新运行的表里：`configs/gate.yaml` 的当前内容就是选定的最终配置，
`ablation` 表逐行写明它覆写了哪些键，`ladder` 表把两条阶梯的取值区间整个留下，`gradientScale-ladder.csv` 留下
这些变体在源参数处的梯度范数与两者之比。

### 4.4 可复现性验证范围

下列每一条都在当前配置与当前代码上实际执行过，读数就是执行打印出来的数，对照对象是执行之前 `results/` 里的
文件——也就是第 3 节引用的那批行：

- 把 `data/` 与 `models/checkpoints/` 同时改名后运行不带参数的命令：冻结配置、主表、流表、等位移表全部照常
  打印，五张图全部重画，读到的每个数都来自 `results/*.csv`，因此第 3 节不需要数据与 checkpoint 就能核对；
- `check`：`oracle` 等价自检四行为 1.803e-04、0.000e+00、0.000e+00、0.000e+00；十个验证格上 40 of 40 个 live
  精度等于 `pilot.csv` 与 `ladder.csv` 的行，max |delta| 0.0000 pp；
- `pilot`：重写 30 行，`accuracy` 与 `meanEntropy` 逐行等于重写前的文件，只有 `seconds` 变化，判定行仍是
  `FAIL -> switch to plan B`，两条读数是 +10.97 pp 与 −2.20 pp；
- `gradient ladder`：重写 26 行，八个数值列逐行相等，λ=16 在 batch 64 上的 `termRatio` 为 45.01；
- `figures`：两轮重画的五张 PNG 与重画前的五份文件 MD5 全部相同；
- `throughput`：MPS 上三次运行的一致性对 TENT 延迟比值落在 1.96–2.00，峰值内存三档都是 67 MB；把 `deviceOrder`
  临时改成 `[cpu]` 跑同一条命令，三行照常打印、内存读整个进程的峰值 RSS；再在屏蔽 `resource` 导入的子进程里跑
  一次，内存一列打印 `n/a`，命令跑完退出，随后配置已改回 `[mps, cuda, cpu]`。Windows 真机上的端到端运行本轮
  没有条件执行，屏蔽导入只证明缺这个模块时不再崩；
- `test`：26 个用例全部通过，退出码 0，其中 `testGateWeights` 打印的两个跨 batch 差值是 0.206 与 0.022。

`main`、`stream`、`gate ablation` 与 `gate ladder` 四段是本研究最贵的运行，本轮没有整段重跑；`results/` 里现在的
一千一百七十行、三百二十四行、四百二十行与五百六十行由当前代码在当前配置下写出。这四段写完之后改动过三处，
每一处都不触及它们的读数：`batchOrder` 的补齐修正只在 `count` 小于一个 batch 时改变输出，实际的 36 个
（张数, batch, seed）组合逐字节相同；`efficiency.py` 只被 `throughput` 使用；`gate.yaml` 本轮只加注释，键值未动。
`check` 不比对带一致性项的那一行，原因是没有一行门控读数跑在 `configs/gate.yaml` 的默认值上——`ablation` 的每个
变体都自己钉住开关，对照对象因此不存在。等位移那条轴的九个读点由不带参数的运行直接从 `ladder.csv` 的
`displacementNorm` 列读出，本轮的打印与 3.3 的表格逐行相同。
