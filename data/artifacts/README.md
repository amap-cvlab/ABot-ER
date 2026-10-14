# ModelScope downloads / 模型与数据下载

ABot-ER models and training data are distributed on **ModelScope**, not Hugging Face.
ABot-ER 模型和训练数据均通过 **ModelScope** 分发，没有对应的 Hugging Face 发布仓库。

## Release repositories / 发布入口

| Artifact / 产物 | ModelScope |
| --- | --- |
| Model collection / 模型合集 | [ABot-ER](https://www.modelscope.cn/collections/amap_cvlab/ABot-ER) |
| 2B model / 模型 | [ABot-ER-2B](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-2B) |
| 4B model / 模型 | [ABot-ER-4B](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-4B) |
| 9B model / 模型 | [ABot-ER-9B](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-9B) |
| 27B model / 模型 | [ABot-ER-27B](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-27B) |
| 35B-A3B model / 模型 | [ABot-ER-35B-A3B](https://www.modelscope.cn/models/amap_cvlab/ABot-ER-35B-A3B) |
| Training data / 预训练与后训练数据 | [ABot-ER-Training-Data](https://www.modelscope.cn/datasets/amap_cvlab/ABot-ER-Training-Data) |

## Download and prepare / 下载与准备

Use the download instructions and file listing on each ModelScope repository page.
Keep the model configuration, tokenizer, processor, weight index, and all weight
shards together. Follow the dataset card for annotations, media, and their directory
layout, then continue with the [reproduction guide](../../docs/reproduction.md).
Benchmark test data is managed separately; it is not training data.

请按各 ModelScope 仓库页面的下载说明获取文件。模型的配置、tokenizer、processor、
权重索引及全部权重分片须配套保留。数据的标注、媒体和目录布局以数据集卡片为准，
准备完成后按照[复现指南](../../docs/reproduction.md)运行。基准测试数据单独管理，不属于训练数据。

## Integrity and reproducibility / 完整性与复现记录

Record the repository ID, resolved revision, download date, file sizes, and SHA-256
hashes. Where the release provides a checksum manifest, compare every required file
against it before training or evaluation. A locally computed hash records downloaded
bytes but does not establish authenticity without a trusted reference. A repository
URL alone is not an immutable snapshot.

记录仓库 ID、实际下载 revision、下载日期、文件大小和 SHA-256。如果发布仓库提供校验清单，
训练和评测前应逐文件核对。自行计算的 hash 只能记录已下载的内容，没有可信的参考值时
不能据此宣称已验证真实性；仓库 URL 本身也不等于不可变快照。

The dataset is pinned by [modelscope_release.json](../modelscope_release.json) to
revision `4afef1c8728fd11abec9a1e443162bd3dd3afe3b`. Download that revision's
`SHA256SUMS.txt`, `RELEASE_MANIFEST.json`, and `data/` directory through ModelScope,
preserving paths. `abot-er verify-public-data` verifies the checksum-list identity
and every shard of the selected stage before creating a local preparation manifest.
It does not download media or prove semantic equivalence to historical training data.

数据下载必须固定上述 revision，而不是可变的 `master`；下载后使用公开版校验入口。
模型仍需记录实际下载 revision 与文件哈希。这里不宣称已完成全量模型下载或 GPU 复现。

## Legacy compatibility / 历史兼容说明

The existing `abot-er download` command and optional `[hub]` extra implement a legacy
Hugging Face-only download path. They are **not** the download instructions for this
ModelScope release; do not pass ModelScope repository IDs to that command. They remain
in the code for compatibility, not as a claim that ABot-ER is hosted on Hugging Face.
References to “Hugging Face-format checkpoints” elsewhere describe a file format,
not the hosting platform.

现有 `abot-er download` 命令与可选 `[hub]` 扩展是历史 Hugging Face 下载实现，
**不适用于本次 ModelScope 发布**，不要向该命令传入 ModelScope 仓库 ID。
为兼容性保留这些代码，不表示模型或数据发布在 Hugging Face。
其他文档中的“Hugging Face 格式 checkpoint”指文件格式，不是托管平台。

Model, dataset, and third-party licenses remain separate from the code license.
Never put access tokens or other credentials in Git or command-line examples.
模型、数据集和第三方内容的许可证与代码许可证分别管理；不要将访问凭据写入 Git 或命令示例。
