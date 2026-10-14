# Data releases

Git 保存训练与评测代码、配方和数据清单，不保存原始训练媒体、测试答案或模型权重。发布数据见 [ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABot-ER-Training-Data)。

## 最终公开版本（ModelScope 为准）

固定 revision：`4afef1c8728fd11abec9a1e443162bd3dd3afe3b`，release 为
`20261013_academic_jsonl_v3`。Stage A 为 **462,385** 条 / 288 shard；Stage B 为
**104,960** 条 / 128 shard；合计 **567,345** 条 / 416 shard。均为物理记录口径。
版本锚点与校验清单自身的 SHA 见 [modelscope_release.json](modelscope_release.json)。
下载、校验和路径准备见[复现说明](../docs/reproduction.md#2-获取并验证模型和数据)。

Git 不重复存放 JSONL。公开版使用其自己的 SHA 清单，不应拿下方历史清单校验。
预训练数据包含 **462,385 条记录**；下载与复现以 ModelScope 固定版本及对应校验清单为准。
使用 `--public-data` 选择公开版本，训练记录会保留其版本身份。原始 ModelScope 文件保持不变。

## 历史训练身份（只作血缘证据）

| 阶段 | ID | 物理记录 | 历史 manifest |
| --- | --- | ---: | --- |
| A | joint_tp_2dqa_r4_clean_dhta_opensource_vabenchxml8_20260921 | 462390 | [pretrain](manifests/pretrain_vabxml8.json) |
| B | joint_tp_2dqa_dhta_final_joint_v1_20260910 | 104960 | [posttrain](manifests/posttrain_final_joint.json) |

Stage A有46,776条VABench监督修复为正式8点XML；真实处理是已验证单点重复8次。其余415,614条与父版字节一致（历史构建报告记录）。这不是“新增了8个独立区域点”。Stage B沿用Final Joint封板release，52,480个identity各曝光两次；不要将物理行数当作独立样本数。

## Stage B实际配比

| 五维能力 | benchmark proxy | 物理记录 |
| --- | --- | ---: |
| Spatial & Relational | VSI / ERQA / EmbSpatial | 4000 / 10496 / 6000 |
| Scene & State | OpenEQA / UniEQA / RoboVQA | 12000 / 5356 / 8000 |
| Temporal & Event | Cosmos | 6074 |
| Point & Affordance | VABench / W2P | 16794 / 10000 |
| Long-Horizon Action Organization | ET / ALFRED / Habitat | 5240 / 12000 / 9000 |

五维合计：20,496 / 25,356 / 6,074 / 26,794 / 26,240。QA:Plan按物理行数为75:25；历史构建报告的assistant监督token则为36.2379%:63.7621%。不要沿用其他release的90.43:9.57。上述是目标桶，不是公开源名或数据许可。

历史训练元数据的缺失字段统计不用于描述当前公开导出包。
公开版组件计数和来源名称以固定 revision 的 `RELEASE_MANIFEST.json` 与
`manifests/source_manifest.csv` 为准；`openeqa` 等代理桶名不表示样本必然来自
该 benchmark 的官方训练集。许可与媒体获取方式以各来源条款和数据卡为准。

## 格式与路由

- ShareGPT `conversations`，媒体复数字段`images`、`videos`；字段映射随dataset_info冻结。
- VABench8点XML与W2P tuple-list独立；坐标均为[0,1]，三位小数。
- 路由在loader注入，EQA为`[2D_QA]`，Plan为`[TASK_PLAN]`；不得把这两个token同时加到样本中。
- ET、ALFRED、Habitat各自保留真实动作schema，不合并动作空间。

## 发布范围与复现

本项目发布经过授权的封板预训练、后训练数据，以及直接使用这些数据的训练与评测代码。
复现以发布的数据版本为起点，无需运行内部数据构建流水线。

模型与数据入口见 [ModelScope 下载指南](artifacts/README.md)。数据版本、分片结构及
JSONL 校验信息以 ModelScope 指定 revision 的 `SHA256SUMS.txt` 为准；
来源与许可、媒体获取方式以发布包说明为准。
校验范围与审计说明见[发布说明](../docs/release.md#evidence-limitations)。
