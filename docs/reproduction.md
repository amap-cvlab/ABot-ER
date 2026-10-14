# Reproduction guide

## 0. 支持状态

本版优先固定Qwen3.5-9B + Task-Conditioned Adaptive Training Strategy的最新封板主线。2B/4B、MoE、27B不应直接套用本配方；其层号和架构需单独迁移、验证。完整训练环境与正式评测仍在验收，不承诺当前任意新环境可运行。

## 1. 获取代码与环境

在仓库根目录使用Python 3.10，执行：

```bash
python -m pip install -e .
python scripts/capture_environment.py > environment.json
python -m unittest discover -s tests -p 'test_reproduction.py' -v
```

目前仅支持源码editable安装；训练框架与configs通过仓库相对路径定位，不能把生成的wheel脱离仓库使用。

GPU训练还需要匹配的PyTorch/Transformers/DeepSpeed/TRL/PEFT/Datasets、视频解码与Qwen3.5线性注意力依赖。请固定容器 digest 和依赖版本，并在所用 GPU 环境验证分布式训练与模型保存重载。

## 2. 获取并验证模型和数据

模型和数据以 ModelScope 最终发布版为准，入口见[下载指南](../data/artifacts/README.md)。
本指南默认使用公开导出数据，训练与直接评测的起点不同：

| 用途 | 获取或生成方式 |
| --- | --- |
| 直接评测发布模型 | 下载 [ABot-ER-9B](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-9B)，按评测指南运行 |
| 从基座运行两阶段训练 | 准备原始 Qwen3.5-9B 基座，运行 Stage A，再以其 `checkpoint-2800` 进入 Stage B |
| Stage A 中间权重 | 本发布不提供下载；由用户按 Stage A 配方自行训练生成 |

原始基座的精确上游 revision 与权重校验清单尚未固定在本仓库中；
请记录实际使用版本，不应将任意同名基座视为已验证的完全相同起点。
最终 Stage B 发布模型不能替代原始基座或 Stage A 初始化。

参考 9B 配方的 Stage A 最大训练步数为 3,600，选取第 2,800 步进入
一次 Stage B 训练；Stage B 最大 800 步、每 100 步保存，报告中的发布模型
选取第 300 步。阶段边界重置优化器与 scheduler。上述数字描述训练配方及
报告模型的选取，不表示运行公开数据版本必然得到逐位相同的权重或分数。

从 ModelScope 下载数据 revision `4afef1c8728fd11abec9a1e443162bd3dd3afe3b`，
保持 `data/`、`SHA256SUMS.txt` 和 `RELEASE_MANIFEST.json` 的相对位置。媒体二进制
不在此 JSONL 包中；按数据卡与媒体清单从上游取得，并保留相对 `assets/...` 目录。

```bash
abot-er verify-public-data --data /your/downloads/ABot-ER-Training-Data \
  --stage stage_a_pretrain --output /your/verified/stage-a.json
abot-er verify-public-data --data /your/downloads/ABot-ER-Training-Data \
  --stage stage_b_posttrain --output /your/verified/stage-b.json
```

该检查先验证校验清单及 release manifest 自身的 SHA，再逐片验证 JSONL SHA、
实际记录数、分片集合与路由字段，输出本地准备清单。它不验证媒体内容、不重跑污染审计。
公开数据身份由 [ModelScope 固定版本清单](../data/modelscope_release.json)约束，
下载后的校验依据是该版本的 `RELEASE_MANIFEST.json` 与 `SHA256SUMS.txt`。
`data/manifests/` 保留的是历史训练身份，不是公开导出版的下载或校验清单；
以下训练命令使用 `--public-data` 显式选择公开版本。

## 3. 迁移媒体路径

ModelScope 下载目录保持不变。在本地创建 `media_roots.json`，将公开相对路径映射
到已取得的本地媒体目录。例如（本地路径示例，不是下载地址）：

```json
{"assets": "/your/datasets/assets"}
```

```bash
abot-er prepare-data --data /your/downloads/ABot-ER-Training-Data/data/stage_a_pretrain \
  --manifest /your/verified/stage-a.json \
  --media-roots media_roots.json --output /your/prepared/pretrain
abot-er prepare-data --data /your/downloads/ABot-ER-Training-Data/data/stage_b_posttrain \
  --manifest /your/verified/stage-b.json \
  --media-roots media_roots.json --output /your/prepared/posttrain
```

只替换媒体路径；不修改prompt/label、行序或分片归属。每个目标媒体路径必须存在且不能逃逸映射根目录。输出包含新字节hash与旧release身份、dataset_info.json和shards。输出目录必须不存在；若失败留下局部文件，不可拿它训练，应检查原因后选新目录重试。当前不复制或重编码媒体，媒体下载和许可由数据发布清单管理。

## 4. 提交前dry-run

```bash
abot-er train --recipe configs/pretrain/q35_9b_dhta.yaml \
  --model /your/models/qwen35-9b-base --data /your/prepared/pretrain \
  --output /your/runs/stage-a --world-size 32 --public-data --dry-run
```

dry-run只检查模型config架构、release身份、GA和分片算术容量，不加载模型、不启动训练。它不是逐rank loader完整消费验证。正式训练前还必须完成分布式smoke与动态过滤统计；后训练仅有20步静态余量，不能忽略任何运行时丢样本。

## 5. Stage A

以下为4节点、每节点8卡的32卡拓扑示例。每个节点以不同`NODE_RANK=0..3`运行同一命令，`MASTER_ADDR`为rank0可达地址；不是将四条命令运行在同一节点。不需要内部集群SDK。

```bash
NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR="${MASTER_ADDR:?Set the rank-0 host address}" MASTER_PORT=29500 \
  bash scripts/train.sh --recipe configs/pretrain/q35_9b_dhta.yaml \
  --model /your/models/qwen35-9b-base --data /your/prepared/pretrain \
  --output /your/runs/stage-a --public-data
```

正式主线固定32卡、GA4、global batch128。其他拓扑即使batch相同也不能直接宣称数值等价。输出为本地HF格式 checkpoint，不依赖内部存储服务。禁止继承外部自动恢复配置，禁止复用非空输出目录。

## 6. Stage B：一次后训练

从本次Stage A的`checkpoint-2800`加载，重新初始化优化器与scheduler；不是从base或历史多轮后训练链继续。

```bash
abot-er train --recipe configs/posttrain/q35_9b_dhta_final_joint.yaml \
  --model /your/runs/stage-a/checkpoint-2800 --data /your/prepared/posttrain \
  --output /your/runs/stage-b --world-size 32 --public-data --dry-run

NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR="${MASTER_ADDR:?Set the rank-0 host address}" MASTER_PORT=29500 \
  bash scripts/train.sh --recipe configs/posttrain/q35_9b_dhta_final_joint.yaml \
  --model /your/runs/stage-a/checkpoint-2800 --data /your/prepared/posttrain \
  --output /your/runs/stage-b --public-data
```

`save_only_model=true`不保存可完整续训的optimizer状态。不可将“从权重开始新SFT”写成“无损恢复中断训练”。输出保留model/config/tokenizer/processor和复现manifest。

## 7. 验证与评测

```bash
python tests/legacy/test_dhta.py
python tests/legacy/test_dhta_qwen35_attention_smoke.py \
  --base-config /your/models/qwen35-9b-base
```

第一个为轻量Task-Conditioned Adaptive Training Strategy算法测试；第二个检查真实Qwen注意力类的注入与state_dict round-trip，但不是整模型GPU训练验证。它为登录节点可选torchaudio ABI问题使用测试内隔离，不是生产模型代码。

正式12项评测请见[运行说明](../evaluation/RUNNING.md)。请在目标环境分别验证 EQA 推理、Judge 和规划模拟器运行。EQA和TP分开运行，不提供掩盖不同依赖和覆盖率的`evaluate-all`命令。

## 8. 可复现性的边界

`--public-data` 显式选择 ModelScope 导出版本；训练记录标注 `modelscope_public_export`。
预训练数据包含 462,385 条记录；下载与复现以 ModelScope 固定版本及对应校验清单为准。
不加此参数时仍严格要求历史清单及原始 SHA，不能静默混用。

固定数据、原始模型revision、代码、容器、种子、GPU拓扑、生成与评分配置。BF16、ZeRO-3和分布式算子不保证逐位权重相同；须报告重复实验逐项分数及格式成功率，不以训练loss接近替代复现验证。所有分数和SOTA结论需要同覆盖率、同评测合同、同judge版本。
