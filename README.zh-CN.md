<div align="center">

# ABot-ER

### 具身理解与任务规划

[![arXiv：2610.18355](docs/assets/arxiv-badge.svg)](https://arxiv.org/abs/2610.18355)
[![PDF 技术报告](docs/assets/report-badge.svg)](https://arxiv.org/pdf/2610.18355)
[![Project Page](docs/assets/project-page-badge.svg)](https://amap-cvlab.github.io/ABot-ER/)
[![License: Apache 2.0](docs/assets/license-badge.svg)](LICENSE)
[![ModelScope 数据集](docs/assets/modelscope-datasets-badge.svg)](https://www.modelscope.cn/datasets/amap_cvlab/ABot-ER-Training-Data)
[![ModelScope Models](docs/assets/modelscope-models-badge.svg)](https://www.modelscope.cn/collections/amap_cvlab/ABot-ER)

[![Hugging Face Datasets](docs/assets/hf-datasets-badge.svg)](https://huggingface.co/datasets/acvlab/ABot-ER-Training-Data)
[![Hugging Face Models](docs/assets/hf-models-badge.svg)](https://huggingface.co/collections/acvlab/abot-er)

ABot-ER 系列模型页面及训练数据已在 Hugging Face 公开；35B-A3B 页面提供 ModelScope 权重下载链接。

[English](README.md) | **简体中文**

[实验结果](#实验结果) · [模型](#模型) · [数据集](#数据集) · [安装](#安装) · [训练](#训练) · [评测](#评测) · [文档](docs/README.md)

</div>

ABot-ER 在共享的视觉语言模型中联合建模具身视觉理解与长程任务规划，覆盖五个能力维度：空间关系、场景状态、时序事件、点定位与可供性，以及动作组织。

本仓库提供 **Task-Conditioned Adaptive Training Strategy（任务条件自适应训练策略）**实现、两阶段训练配置、版本化训练数据清单，以及九项具身问答和三项任务规划基准的评测接口。

模型与训练数据可通过上方 Hugging Face 标签及[模型](#模型)、[数据集](#数据集)中的 ModelScope 链接获取。预训练与后训练数据统一发布于同一个 Hugging Face 训练数据仓库。

## 项目概览

<p align="center">
  <a href="docs/assets/paper/capability-overview.png"><img src="docs/assets/paper/capability-overview.png" width="100%" alt="ABot-ER 将观测与指令映射到语义回答、动作计划和空间目标，图中展示五类能力示例。"></a>
</p>

*论文图 2：共享推理模型覆盖五维能力，输出回答、计划与坐标；动作执行由智能体运行时和下游策略完成。图中的示例来自评测集，不是训练样本。*

- **共享模型与任务条件适配。** 该策略在指定注意力投影上引入任务条件低秩残差，同时保留可训练的共享语言模型主干。
- **两阶段训练。** 联合领域监督微调建立基础能力覆盖，随后使用封板联合后训练数据执行一次连续监督微调。
- **可追溯的数据与 checkpoint。** 训练配置关联不可变数据清单、分片校验值及明确的 checkpoint 血缘。
- **基准独立的评测协议。** 评测接口保留原始提示词、解析器、评分器和任务路由行为。

| 能力维度 | 评测基准 |
| --- | --- |
| 空间与关系推理 | EmbSpatial、VSI、ERQA |
| 场景与状态理解 | OpenEQA、UniEQA、RoboVQA |
| 时序与事件推理 | Cosmos |
| 点定位与可供性 | VABench-P、Where2Place |
| 长程动作组织 | ET-Plan-Bench、EB-ALFRED、EB-Habitat |

### 模型架构

<p align="center">
  <a href="docs/assets/paper/model-architecture.png"><img src="docs/assets/paper/model-architecture.png" width="100%" alt="9B 模型架构：共享多模态解码器、任务条件 Q/V 残差、确定性的问答与规划路由，以及语义、点和计划输出。"></a>
</p>

*论文图 3：9B 实现在第 19、23、27、31 层（从 0 编号）的 Q/V 投影上加入任务条件残差，每条序列选择一条确定性路由。图中 EQA 标签对应代码中的 `[2D_QA]`，规划使用 `[TASK_PLAN]`。点击图片可查看高清原图。*

## 实验结果

<p align="center">
  <a href="docs/assets/paper/benchmark-overview.png"><img src="docs/assets/paper/benchmark-overview.png" width="100%" alt="论文图 1：十二项基准雷达对比与感知—规划散点图；气泡面积表示执行定位均分。"></a>
</p>

*论文图 1：左侧为十二项基准对比；右侧以感知、规划均分为坐标，气泡面积表示执行定位均分。比较范围为下表九个基线；雷达图面积不是综合分数。*

### 主结果 — ABot-ER 9B

结果对应[技术报告](https://arxiv.org/abs/2610.18355)表 8。

**12 项中 6 项超过所列基线** · **EQA 64.57** · **Planning 41.61**

分数以 0–100 展示，越高越好。完整十二项基准及所列基线结果见下方可展开的对比表。

<details>
<summary><strong>展开完整对比 · 全部 10 个模型、五个能力维度</strong></summary>

所有模型均有完整十二项结果。**粗体**表示该基准或汇总指标的最高分；`--` 为有效零分，计入均值。

#### 空间与关系

| Model | EmbSpatial | VSI | ERQA |
| --- | ---: | ---: | ---: |
| **ABot-ER 9B** | 81.68 | 63.30 | **47.31** |
| Cosmos Reason 7B | 66.96 | 27.41 | 40.28 |
| EmbodiedBrain 7B | 74.37 | 37.05 | 38.27 |
| Cambrian-S 7B | 69.13 | 63.39 | 35.23 |
| IflyBot-VLM 8B | 75.05 | 30.47 | 38.38 |
| Mimo-Embodied 7B | 77.59 | 43.86 | 39.50 |
| RynnBrain 8B | 81.46 | 67.55 | 40.40 |
| Hy-Emb. RxBrain 1.0 | 80.38 | 0.16 | 38.58 |
| RynnBrain 1.1 9B | **82.53** | **71.59** | 44.02 |
| Cosmos3 Nano | 82.03 | 48.65 | 46.09 |

#### 场景与状态

| Model | OpenEQA | UniEQA | RoboVQA |
| --- | ---: | ---: | ---: |
| **ABot-ER 9B** | 63.33 | 50.66 | **71.79** |
| Cosmos Reason 7B | 35.17 | 38.38 | 53.75 |
| EmbodiedBrain 7B | 51.79 | 39.02 | 63.19 |
| Cambrian-S 7B | 53.19 | 43.50 | 37.77 |
| IflyBot-VLM 8B | 52.95 | 40.63 | 53.60 |
| Mimo-Embodied 7B | **65.85** | 43.93 | 66.09 |
| RynnBrain 8B | 57.06 | 49.68 | 64.31 |
| Hy-Emb. RxBrain 1.0 | 40.90 | 39.51 | 19.66 |
| RynnBrain 1.1 9B | 61.60 | **56.62** | 50.63 |
| Cosmos3 Nano | 49.22 | 49.57 | 56.08 |

#### 时序与事件

| Model | Cosmos |
| --- | ---: |
| **ABot-ER 9B** | **68.96** |
| Cosmos Reason 7B | 51.13 |
| EmbodiedBrain 7B | 46.68 |
| Cambrian-S 7B | 44.61 |
| IflyBot-VLM 8B | 54.95 |
| Mimo-Embodied 7B | 50.27 |
| RynnBrain 8B | 54.84 |
| Hy-Emb. RxBrain 1.0 | 35.89 |
| RynnBrain 1.1 9B | 52.42 |
| Cosmos3 Nano | 63.18 |

#### 任务规划

| Model | ET-Plan | EB-ALFRED | EB-Habitat |
| --- | ---: | ---: | ---: |
| **ABot-ER 9B** | 61.15 | **26.67** | **37.00** |
| Cosmos Reason 7B | 1.32 | 2.00 | 17.00 |
| EmbodiedBrain 7B | 65.89 | 0.33 | 8.67 |
| Cambrian-S 7B | 39.35 | -- | 3.00 |
| IflyBot-VLM 8B | 30.75 | 0.33 | 4.67 |
| Mimo-Embodied 7B | 68.17 | 17.33 | 21.00 |
| RynnBrain 8B | 66.39 | 0.33 | 13.00 |
| Hy-Emb. RxBrain 1.0 | 20.02 | -- | -- |
| RynnBrain 1.1 9B | 56.35 | 5.00 | 10.33 |
| Cosmos3 Nano | **69.55** | 16.00 | 26.67 |

#### 点定位与可供性

| Model | VABench-P | Where2Place |
| --- | ---: | ---: |
| **ABot-ER 9B** | **63.33** | 70.79 |
| Cosmos Reason 7B | 4.33 | 5.21 |
| EmbodiedBrain 7B | 2.58 | 34.45 |
| Cambrian-S 7B | 0.50 | 10.94 |
| IflyBot-VLM 8B | 11.08 | 70.76 |
| Mimo-Embodied 7B | 46.90 | 63.60 |
| RynnBrain 8B | 9.00 | **75.48** |
| Hy-Emb. RxBrain 1.0 | 3.58 | 40.50 |
| RynnBrain 1.1 9B | 8.38 | 70.20 |
| Cosmos3 Nano | 47.54 | 57.68 |

#### 汇总分数

| Model | Perception (7) | Planning (3) | Execution (2) |
| --- | ---: | ---: | ---: |
| **ABot-ER 9B** | **63.86** | **41.61** | **67.06** |
| Cosmos Reason 7B | 44.73 | 6.77 | 4.77 |
| EmbodiedBrain 7B | 50.05 | 24.96 | 18.52 |
| Cambrian-S 7B | 49.55 | 14.12 | 5.72 |
| IflyBot-VLM 8B | 49.43 | 11.92 | 40.92 |
| Mimo-Embodied 7B | 55.30 | 35.50 | 55.25 |
| RynnBrain 8B | 59.33 | 26.57 | 42.24 |
| Hy-Emb. RxBrain 1.0 | 36.44 | 6.67 | 22.04 |
| RynnBrain 1.1 9B | 59.92 | 23.89 | 39.29 |
| Cosmos3 Nano | 56.40 | 37.41 | 52.61 |

| Model | EQA (9) | All (12) |
| --- | ---: | ---: |
| **ABot-ER 9B** | **64.57** | **58.83** |
| Cosmos Reason 7B | 35.85 | 28.58 |
| EmbodiedBrain 7B | 43.04 | 38.52 |
| Cambrian-S 7B | 39.81 | 33.38 |
| IflyBot-VLM 8B | 47.54 | 38.63 |
| Mimo-Embodied 7B | 55.29 | 50.34 |
| RynnBrain 8B | 55.53 | 48.29 |
| Hy-Emb. RxBrain 1.0 | 33.24 | 26.60 |
| RynnBrain 1.1 9B | 55.33 | 47.47 |
| Cosmos3 Nano | 55.56 | 51.02 |

</details>

**汇总口径。** Perception (7) 包含空间、场景和时序基准；Execution (2) 包含 VABench-P 与 Where2Place，两者共同构成 EQA (9)。Planning (3) 包含 ET-Plan、EB-ALFRED 和 EB-Habitat。各均值对所属基准等权计算。All (12) 为十二项直接平均，**不是 EQA (9) 与 Planning (3) 的二者平均**；它混合了评分型和成功率指标，应结合单项结果解读。

### 规模对比 — 单次后训练配方

结果对应[技术报告](https://arxiv.org/abs/2610.18355)表 10。

所有模型使用**相同两份数据版本，分别执行一次 Stage A 和一次 Stage B**，各自选取最优 checkpoint。粗体为各规模中的最高分。此处展示报告实验结果，不代表仓库独立复现验证已全部完成。

| Model | EQA (9) | Planning (3) | All (12) |
| --- | ---: | ---: | ---: |
| 2B | 55.20 | 34.14 | 49.93 |
| 4B | 61.80 | 41.69 | 56.77 |
| 9B | 64.57 | 41.61 | 58.83 |
| 27B | **66.23** | **51.59** | **62.57** |
| 35B-A3B | 65.39 | 46.48 | 60.66 |

<details>
<summary><strong>展开完整规模对比 · 十二项基准与能力均值</strong></summary>

#### 空间与关系

| Benchmark ↑ | 2B | 4B | 9B | 27B | 35B-A3B |
| --- | ---: | ---: | ---: | ---: | ---: |
| EmbSpatial | 78.87 | 79.53 | 81.68 | 82.39 | **82.69** |
| VSI | 49.85 | 58.85 | 63.30 | 65.62 | **66.85** |
| ERQA | 36.93 | 40.36 | 47.31 | **49.65** | 47.84 |

#### 场景与状态

| Benchmark ↑ | 2B | 4B | 9B | 27B | 35B-A3B |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenEQA | 56.21 | 60.10 | 63.33 | **67.25** | 65.33 |
| UniEQA | 44.43 | 48.50 | 50.66 | **53.45** | 52.60 |
| RoboVQA | 68.64 | 72.32 | 71.79 | 73.39 | **74.22** |

#### 时序与事件

| Benchmark ↑ | 2B | 4B | 9B | 27B | 35B-A3B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cosmos | 55.58 | 68.69 | **68.96** | 67.71 | 65.73 |

#### 任务规划

| Benchmark ↑ | 2B | 4B | 9B | 27B | 35B-A3B |
| --- | ---: | ---: | ---: | ---: | ---: |
| ET-Plan | 64.41 | 60.75 | 61.15 | **69.11** | 67.77 |
| EB-ALFRED | 13.67 | 30.00 | 26.67 | **37.00** | 31.00 |
| EB-Habitat | 24.33 | 34.33 | 37.00 | **48.67** | 40.67 |

#### 点定位与可供性

| Benchmark ↑ | 2B | 4B | 9B | 27B | 35B-A3B |
| --- | ---: | ---: | ---: | ---: | ---: |
| VABench-P | 54.00 | 61.33 | **63.33** | 62.67 | 62.33 |
| Where2Place | 52.26 | 66.47 | 70.79 | **73.90** | 70.91 |

#### 能力均值

| Model | Perception (7) | Execution (2) |
| --- | ---: | ---: |
| 2B | 55.79 | 53.13 |
| 4B | 61.19 | 63.90 |
| 9B | 63.86 | 67.06 |
| 27B | **65.64** | **68.29** |
| 35B-A3B | 65.04 | 66.62 |

</details>

*数值说明：均值保留报告提供值。4B 的九项展示分直接平均为 61.79，而报告中的 EQA (9) 为 61.80；此 0.01 差异需用未四舍五入的源分数核对。*

### 真机演示

<details>
<summary><strong>展开 Galaxea R1 Pro 上的定性演示</strong></summary>

<p align="center">
  <a href="docs/assets/paper/real-world-demonstrations.jpg"><img src="docs/assets/paper/real-world-demonstrations.jpg" width="100%" alt="Galaxea R1 Pro 真机演示：识别积木算式 6 加 9 与结果 15，以及判断有效的布手势。"></a>
</p>

*论文图 7：Galaxea R1 Pro 上的积木算术与石头剪刀布演示。ABot-ER 向 ABot-AgentOS 提供视觉状态信息，由后者协调决策与下游动作。这些是定性演示，不是新增的基准成功率结果。*

</details>

## 模型

模型合集：[ModelScope · ABot-ER](https://www.modelscope.cn/collections/amap_cvlab/ABot-ER)。下表统一展示用于具身问答与任务规划的发布版本；权重文件大小不代表推理或训练所需显存。

| 模型 | 权重大小 | ModelScope |
| --- | --- | --- |
| ABot-ER-2B | 4.45 GB | [查看模型](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-2B) |
| ABot-ER-4B | 9.10 GB | [查看模型](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-4B) |
| ABot-ER-9B | 18.84 GB | [查看模型](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-9B) |
| ABot-ER-27B | 54.74 GB | [查看模型](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-27B) |
| ABot-ER-35B-A3B | 70.25 GB | [查看模型](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-35B-A3B) |

加载时使用各模型配套的 config、tokenizer、processor、路由 token ID 和任务条件残差权重。本仓库参考训练配置面向 **Qwen3.5-9B + Task-Conditioned Adaptive Training Strategy**，不涵盖其他参数规模或模型架构。训练阶段、超参数和 checkpoint 选择说明见[训练指南](docs/reproduction.md)。

## 数据集

预训练数据包含 **462,385 条记录**；下载与复现以 ModelScope 固定版本及对应校验清单为准。使用公开数据不需要内部数据构造流水线。

| 数据集 | 阶段 | 物理记录数 | 分片数 | 数据清单 |
| --- | --- | ---: | ---: | --- |
| 联合预训练数据集 | A | 462,385 | 288 | [固定版本的公开清单](data/modelscope_release.json) |
| 联合后训练数据集 | B | 104,960 | 128 | [固定版本的公开清单](data/modelscope_release.json) |

物理记录数统计训练曝光次数，不一定等于唯一样本数。Stage B 包含 52,480 个唯一 identity，每个曝光两次。[数据指南](data/README.md)记录了来源元数据、能力配比、媒体组织、输出格式和审计覆盖范围。

**训练数据：** [ModelScope · ABot-ER-Training-Data](https://www.modelscope.cn/datasets/amap_cvlab/ABot-ER-Training-Data)。文件内容与获取方式以数据集卡片为准，revision 与完整性要求见[产物下载指南](data/artifacts/README.md)。基准测试数据单独管理，仅用于评测。

## 安装

使用 **Python 3.10**，在仓库源码根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
abot-er --help
```

以上命令安装仓库 CLI。模型和数据使用上方 ModelScope 入口获取，此流程不需要安装 Hugging Face 下载扩展。GPU 训练与评测还需要兼容的 PyTorch、Transformers、注意力算子及相关后端依赖。经过验证的 GPU 环境锁定文件仍在验收中；以上命令**不构成完整的 GPU 环境安装**。

环境要求与数据准备步骤见[安装及复现指南](docs/reproduction.md)。可使用以下命令运行轻量数据与训练配置检查：

```bash
python -m unittest discover -s tests -p 'test_reproduction.py' -v
```

## 训练

参考流程从 Qwen3.5-9B 基模型开始，选取 Stage A 的 `checkpoint-2800`，再执行一次 Stage B 训练。这里的“预训练”指在已有视觉语言模型上进行领域监督适配，而非从随机初始化开始训练。

| 配置项 | Stage A | Stage B |
| --- | --- | --- |
| 训练配置 | [联合预训练](configs/pretrain/q35_9b_dhta.yaml) | [联合后训练](configs/posttrain/q35_9b_dhta_final_joint.yaml) |
| 初始化 | Qwen3.5-9B 基模型 | Stage A `checkpoint-2800` |
| 学习率 | `4e-6` | `3e-7` |
| 最大训练步数 | 3,600 | 800 |
| 保存间隔 | 400 步 | 100 步 |
| 全局 batch size | 128 | 128 |
| 参考硬件规模 | 32 张 GPU | 32 张 GPU |

两个阶段均使用 BF16、DeepSpeed ZeRO-3、单卡 micro-batch size 1、梯度累积 4，以及 8,192 的最大序列长度。视觉编码器主体冻结，连接器、语言模型和 任务条件残差参与训练。Stage B 使用新初始化的优化器和学习率调度器。

完成[数据校验与媒体路径重定位](docs/reproduction.md#3-迁移媒体路径)后，可先检查配置而不启动训练：

```bash
abot-er train \
  --recipe configs/pretrain/q35_9b_dhta.yaml \
  --model /models/qwen35-9b-base \
  --data /data/abot-er/pretrain-prepared \
  --output /runs/abot-er/stage-a \
  --world-size 32 --public-data --dry-run
```

上述路径为本地路径示例，不是下载地址。Dry run 检查配置和静态分片容量，不执行分布式数据加载。多节点启动命令、阶段间 checkpoint 衔接及完整超参数见[两阶段复现指南](docs/reproduction.md)。

训练会将 Hugging Face 格式的 checkpoint 和复现清单保存到本地。可移植入口不依赖内部集群服务或 MOS 凭据。

## 评测

评测分为具身问答和交互式任务规划两个环境，分别使用冻结的基准配置，并将运行元数据与原始逐样本输出一同保存。

### 具身问答

根据 [datasets.example.json](configs/eval/datasets.example.json)准备本地数据路径清单。安装好评测依赖、模型和基准数据后，可运行小规模点定位检查：

```bash
abot-er evaluate-eqa \
  --model /models/abot-er \
  --datasets /data/eqa-manifest.json \
  --output /runs/abot-er/eqa-smoke \
  --tasks vabench_aboter,where2place-point_aboter \
  --limit 2 --batch-size 1 --no-judge
```

这是 smoke test，不是九项基准的正式结果。OpenEQA、RoboVQA 和 UniEQA 的完整评测还需要冻结版本的本地 Judge 模型。完整命令与覆盖率检查见 [EQA 评测指南](evaluation/RUNNING.md#eqa-nine-benchmarks-thirteen-task-configurations)。

### 任务规划

ET-Plan-Bench、EB-ALFRED 和 EB-Habitat 需要各自的模拟器资源及推理服务。评测流程会创建独立运行目录，以及经过校验、仅用于任务规划的模型副本，不修改原始 checkpoint。

按照[任务规划评测指南](evaluation/RUNNING.md#task-planning-official-interactive-agent-and-simulators)准备资源、生成模型副本、启动推理服务并运行各项评测。

### 评测约定

- **任务路由：** 问答输入使用 `[2D_QA]`，规划输入使用 `[TASK_PLAN]`。路由由适配器注入，不要手工重复添加。
- **点坐标输出：** VABench-P 使用 XML 包装的八个归一化点；Where2Place 使用归一化 tuple list。两者分别遵循各自协议。
- **结果报告：** 九项 EQA 均值和三项任务规划均值分开报告；比较结果时，评测覆盖范围与版本必须一致。
- **失败记录：** 保留原始输出，区分缺失结果、生成失败、Judge 失败和有效零分。

详细协议及当前验证范围见[评测指南](evaluation/README.md)。

## 文档

| 我想要…… | 对应指南 |
| --- | --- |
| 复现两个训练阶段 | [安装与训练](docs/reproduction.md) |
| 查看数据配比、格式和来源 | [数据集文档](data/README.md) |
| ModelScope 下载与完整性检查 | [模型与数据下载指南](data/artifacts/README.md) |
| 评测 checkpoint | [评测运行命令](evaluation/RUNNING.md) |

仓库结构与实现边界见[架构说明](docs/architecture.md)。[文档索引](docs/README.md)还列出了维护与发布相关参考。

## 许可证

项目代码沿用本 GitHub 仓库已选定的 [Apache-2.0 许可证](LICENSE)。模型权重、数据集、模拟器资源及第三方代码分别遵循各自许可证，相关许可记录将随对应产物一同发布。

署名及各组件要求见[第三方声明](THIRD_PARTY_NOTICES.md)。

## 致谢

ABot-ER 基于 Qwen 模型系列、LLaMA-Factory/TuningFactory，以及[第三方声明](THIRD_PARTY_NOTICES.md)中列出的基准实现构建。感谢相关模型、数据集和评测框架的维护者。

## 引用

```bibtex
@misc{ding2026aboterunifiedembodiedreasoning,
  title={ABot-ER: Unified Embodied Reasoning through Agentic Data-Model Co-Evolution},
  author={Qiuyu Ding and Heng-Da Xu and Changda Xia and Jiajun Shen and Zihao Guan and Zhengchen Yuan and Jianfang Zhang and Di Yang and Jiayi Tian and Yuting Xu and Fei Wang and Shiao Liu and Wei Zhang and Xiaolong Wu and Wenbin Tang and Zedong Chu and Mu Xu},
  year={2026},
  eprint={2610.18355},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.18355},
}
```
