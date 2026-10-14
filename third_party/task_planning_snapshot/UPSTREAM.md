# 上游版本基线

两个评测框架已 vendoring 到根仓库统一管理，不再保留嵌套 `.git` 或 submodule。后续同步上游时，以本文件中的 commit 为三方合并基线，并保留各自 `CHANGELOG.md` 中记录的本地兼容修改。

| 目录 | 上游 | vendoring 基线 |
|---|---|---|
| `ET-Plan-Bench/` | https://github.com/ET-Plan-Bench/ET-Plan-Bench.git | `f1992118db629fb508730bd72793085c9d8e4831` |
| `EmbodiedBench/` | https://github.com/EmbodiedBench/EmbodiedBench.git | `9be4e980e9cd6bcb38373cd4aab7c32724bdd401` |

原嵌套仓库元数据不属于源码交付物，不提交到远程仓库。
