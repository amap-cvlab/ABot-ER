# CHANGELOG.md

本文档记录本项目对官方 EmbodiedBench 框架的本地必要修改。范围仅限 Embodied Arena Task Planning 子榜单中的 `EB-ALFRED` 和 `EB-Habitat`。

维护原则：

- 优先通过项目根目录 wrapper、环境变量和外部配置适配运行环境。
- 只有框架内部初始化、模型调用或 headless 运行无法在外部处理时，才修改本目录源码。
- 后续修改 `EmbodiedBench/` 内源码、运行配置或数据入口时，必须同步更新本文档。

## 2026-08-29 DHTA Task-Planning 路由注入

涉及文件：

- `embodiedbench/planner/remote_model.py`

修改内容：

- 仅当外部评测 wrapper 设置 `TP_DHTA_TASK_PLAN=1` 时，在 OpenAI-compatible 请求应用 chat template 之前，为第一条 user 消息注入且校验唯一 `[TASK_PLAN]` route token。
- 多模态消息将 route token 文本段放在图像段之前；普通文本消息直接在正文前添加 token。

原因和影响：

- DHTA checkpoint 训练时要求每个输入包含一个明确的任务路由 token；缺少 token 或放在图像占位符之后都与训练输入不一致。
- 标准 Qwen3.5 checkpoint 不设置该环境变量，原消息内容与调用路径完全不变；修改不涉及 evaluator、环境交互或评分逻辑。

## 2026-07-15 运行产物最小化

涉及文件：

- `embodiedbench/envs/eb_alfred/EBAlfEnv.py`
- `embodiedbench/envs/eb_habitat/EBHabEnv.py`
- `embodiedbench/evaluator/eb_alfred_evaluator.py`
- `embodiedbench/evaluator/eb_habitat_evaluator.py`
- `embodiedbench/planner/vlm_planner.py`

修改内容：

- ALFRED/Habitat 视觉输入改为每个 shard 一张滚动临时图片，不再按 episode/step 保存历史 PNG。
- 模型逐步输出仅保留错误分析和动作复现需要的 prompt、原始输出、解析动作与解析错误计数，删除逐步重复的模型配置、时间和 episode 元数据。

原因和影响：

- 图片只用于紧随其后的模型请求，历史帧不参与评分；episode 指令和指标已由官方 result JSON 保存。
- 不改变模型输入、动作执行、episode 指标或官方汇总逻辑，只减少节点临时空间和共享存储产物。

## 2026-07-15 EB-ALFRED episode 异常收敛

涉及文件：

- `embodiedbench/evaluator/eb_alfred_evaluator.py`

修改内容：

- EB-ALFRED evaluator 捕获单个 episode 运行时异常时，记录一次失败 action、失败 reward 和当前 episode 指标后结束该 episode，不再无限 `sleep` 后重试同一个异常点。
- EB-ALFRED evaluator 增加单 episode wall-clock 预算，默认 600 秒，可通过 `EB_ALFRED_EPISODE_TIMEOUT_SEC` 覆盖；超时 episode 按失败样本记录并进入后续 episode。
- planner 或模型服务调用异常与动作执行异常分开处理：前者作为评测系统故障让 shard 显式失败，后者作为模型动作失败结束当前 episode 并记分。
- shard 系统失败恢复时可通过 `EB_REUSE_EXISTING_EPISODES=1` 复用已落盘 episode，避免再次请求模型并覆盖已有结果；正式新评测默认关闭。

原因和影响：

- 官方实现的 `except Exception` 只打印异常并等待 30 秒；当 AI2-THOR/Unity 在某个动作上持续抛异常时，episode step 不推进，最终只能由 shard 级 `timeout` 杀掉整个 shard，导致全量评测无法完成。
- 某些 ALFRED long-horizon 失败 episode 会在官方 30 step budget 内反复执行耗时很长的无效环境动作，单 episode 即可耗尽 shard timeout。
- 这些保护只影响异常或超长耗时路径，不改变正常动作执行、episode step/invalid action budget、task success、task progress、reward 计算或 summary 算分；异常/超时 episode 按失败样本进入原有汇总。

## 2026-07-15 EB-Habitat 系统异常显式失败

涉及文件：

- `embodiedbench/evaluator/eb_habitat_evaluator.py`

修改内容：

- 模型 JSON 格式错误、空 plan 和越界 action 继续沿用官方 `-1`/`-2` 路径，按失败 episode 正常落盘和计分。
- evaluator 或 simulator 抛出的未处理运行时异常不再被无限捕获并每次等待 30 秒，而是记录 traceback 后让当前 shard 显式失败。

原因和影响：

- 原实现会掩盖系统异常并反复执行同一个失败点，最终由 shard timeout 杀进程，造成耗时且原因不明的整片重试。
- 修改后模型输出错误不会触发重试；只有评测系统未成功运行的 shard 才进入显式失败状态，便于基于日志诊断后选择性恢复。

## 2026-07-14 EB-ALFRED 导航整体超时保护

涉及文件：

- `embodiedbench/envs/eb_alfred/thor_connector.py`

修改内容：

- `nav_obj` 的多视角 `TeleportFull` 搜索增加单次导航整体预算，默认 120 秒，可通过 `EB_ALFRED_NAV_TIMEOUT_SEC` 覆盖。
- 超过预算时按一次失败的 `find` 动作返回，由官方 evaluator 按既有 invalid-action、replanning 和 episode-step 逻辑继续处理。

原因和影响：

- 官方实现最多搜索 20 个可达位置、每个位置尝试 5 个 horizon；不可见或不可达对象可能让一次 `find` 超过整个 shard 超时并在 retry 中确定性复现。
- 该保护不修改动作空间、episode/invalid-action budget、任务成功判定、reward 或评分逻辑；已进入的单次 AI2-THOR controller 调用仍不能被协作式检查中断。

## 2026-07-10 vLLM 自定义模型名适配

涉及文件：

- `embodiedbench/planner/remote_model.py`
- `embodiedbench/planner/vlm_planner.py`

修改内容：

- 当 `remote_url` / `OPENAI_BASE_URL` / `OPENAI_API_BASE` 已设置时，`remote` 模型允许任意 served model name 走 OpenAI-compatible `_call_gpt` 路径，不再因为名字不含 `qwen`、`gpt`、`InternVL` 等硬编码片段而抛出 `Unsupported model name`。
- `VLMPlanner` 在 OpenAI-compatible remote endpoint 场景下同样追加既有 JSON 输出模板，覆盖用户自定义的模型服务名。

原因和影响：

- 用户自定义模型服务名不一定包含上游框架硬编码的模型家族关键词，需避免错误触发 `Unsupported model name`。
- 该改动只影响 remote 模型调用路由和 prompt 中既有 JSON 格式提示是否追加，不修改 EB-ALFRED / EB-Habitat 的 episode、动作空间、环境交互、解析逻辑或评分逻辑。
- 未设置 OpenAI-compatible endpoint 的未知模型名仍会报错，避免误吞真正未配置的后端。

## 2026-07-07 评测口径与 wrapper 收敛

涉及文件：

- `embodiedbench/envs/eb_alfred/EBAlfEnv.py`
- `embodiedbench/evaluator/eb_alfred_evaluator.py`
- `embodiedbench/envs/eb_habitat/EBHabEnv.py`
- `embodiedbench/evaluator/eb_habitat_evaluator.py`

修改内容：

- EB-ALFRED 的 `max_episode_steps`、`max_invalid_actions` 改为由 evaluator 显式传入 `EBAlfEnv`，默认保持官方原值 `30/10`。
- EB-Habitat 的 `max_episodes`、`max_episode_steps`、`max_invalid_actions` 改为由 evaluator 显式传入 `EBHabEnv`，默认保持官方原值 `30/10`，不再依赖隐藏环境变量控制正式评测分片和预算。
- Habitat shard 仍按 `start_epi_index + max_episodes` 形成 `[start, start + size)` episode 窗口，但参数现在会进入 Hydra config/config.txt，便于审计。

原因和影响：

- 正式评测固定使用 EmbodiedBench 官方 episode budget；wrapper 不再提供自定义 step/invalid action 预算入口。
- 该改动不修改 task success、task progress、reward、动作空间、任务数据或 simulator step 逻辑。
- wrapper 只负责启动、分片、API 环境变量、并发、日志和结果归档；运行结果统一按官方 `30/10` budget 解释。
- EB 正式 wrapper 显式设置 `EB_DISABLE_RESPONSE_FORMAT=0`，确保 remote 模型调用走 EmbodiedBench 现有 `response_format=json_schema` 路径；这不是 JSON 修复或 scorer 改写，只是避免外部环境变量误关 schema 约束。
- `VLMPlanner` 增加只读模型输出记录，EB-ALFRED/EB-Habitat evaluator 在每个 episode 结束时写出 `model_outputs/episode_*_model_outputs.jsonl`。记录只包含文本 prompt、模型 raw output、解析出的 action 和元数据，不保存图片内容；该日志不参与 action 执行、task success 判定或 summary 算分。

## 2026-07-01 AMD/ROCm 本机适配

### EB-ALFRED headless 运行

涉及文件：

- `embodiedbench/envs/eb_alfred/EBAlfEnv.py`
- `embodiedbench/envs/eb_alfred/thor_connector.py`

修改内容：

- `X_DISPLAY` 支持通过 `EB_ALFRED_X_DISPLAY` 设置，便于 wrapper 指定 Xvfb display。
- episode step 和 invalid action 上限由 evaluator 通过 `max_episode_steps`、`max_invalid_actions` 显式传入 Env。
- UbuntuMono 字体缺失时 fallback 到 Pillow 默认字体，避免因可视化标注字体缺失导致初始化失败。

原因和影响：

- 本机通过 Xvfb headless 运行 AI2-THOR，需要外部脚本控制 display 和 smoke test 运行规模。
- 默认值保持官方原值：`X_DISPLAY=1`、`max_episode_steps=30`、`max_invalid_actions=10`。
- 不修改 ALFRED 的任务成功判定、reward 计算或动作执行逻辑。

### EB-Habitat smoke 规模控制

涉及文件：

- `embodiedbench/envs/eb_habitat/EBHabEnv.py`
- `data`
- `embodiedbench/envs/eb_habitat/data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json`

修改内容：

- 支持通过 evaluator 参数 `max_episodes` 限制 episode 数。
- episode step 和 invalid action 上限由 evaluator 通过 `max_episode_steps`、`max_invalid_actions` 显式传入 Env。
- `max_episodes` 与 `start_epi_index` 组合时按 `[start_epi_index, start_epi_index + max_episodes)` 作为评测窗口，支持全量评测并发分片。
- 仓库根目录运行时创建相对软链接（不纳入 Git）：
  ```text
  data -> embodiedbench/envs/eb_habitat/data
  ```
- 移除 ReplicaCAD scene dataset 配置中 `v3_sc4_staging_*` 的 navmesh 引用；本地 ReplicaCAD 数据目录只包含 `v3_sc0` 到 `v3_sc3` 的 scene/stage/navmesh 文件，保留 sc4 坏引用会导致 Habitat-Sim 初始化时查找不存在的 navmesh 并崩溃。

原因和影响：

- EB-Habitat 在 AMD/Mesa 软件渲染环境下启动成本较高，smoke test 需要可控地缩小运行规模。
- 根目录 `data` 是 EmbodiedBench/Habitat 运行时常见数据入口；统一入口自动创建该链接，不把链接或大体积 assets 纳入 Git。
- 未显式设置参数时，step 和 invalid action 默认值保持官方原值 `30` 和 `10`。
- 未设置 `start_epi_index` 或 `max_episodes` 时仍保持原来的顺序全量评测行为。
- ReplicaCAD 配置修复只删除本地数据中不存在的 navmesh 条目，不修改 EB-Habitat 的任务数据、episode 顺序、动作空间、predicate success 或 reward 计算。
- 不修改 Habitat predicate success、reward、动作空间或任务数据。

### remote 模型接口适配

涉及文件：

- `embodiedbench/planner/remote_model.py`
- `embodiedbench/planner/manip_planner.py`
- `embodiedbench/planner/vlm_planner.py`

修改内容：

- 移除顶层 `lmdeploy` 强依赖导入，改为仅在 `model_type == "local"` 时延迟导入。
- `remote_url` 依次支持：
  ```text
  remote_url
  OPENAI_BASE_URL
  OPENAI_API_BASE
  ```
- OpenAI-compatible 调用显式读取 `OPENAI_API_KEY`。
- `qwen` 类模型在设置 `remote_url` 时可走自托管 OpenAI-compatible 接口；未设置时仍走原 DashScope 路径。
- remote 调用的最大输出 token 数支持通过 `EB_MAX_COMPLETION_TOKENS` 覆盖，默认值从官方 `4096` 调整为 `512`。
- 全量 EB 评测中，remote 调用尊重 wrapper 传入的 `EB_MAX_COMPLETION_TOKENS`；当前设置为 `256`，在避免 vLLM `--max-model-len=4096` 下 ALFRED 长 prompt 触发上下文长度 400 错误的同时，比 `96` 更不容易截断动作 JSON。
- `_call_gpt` 对 OpenAI-compatible 服务返回的上下文长度 400 错误增加自适应降输出 token 重试：`256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1`，只处理 context-length 类错误，其他 API 错误仍直接抛出。
- `_call_gpt` 远程调用支持通过 `EB_DISABLE_RESPONSE_FORMAT=1` 不向 OpenAI-compatible 服务传 `response_format`。
- planner 中对 Qwen/InternVL/Claude 类模型追加 JSON 输出模板的判断改为大小写不敏感，并覆盖 `qwen35-vl` 这类本地 served model name。

原因和影响：

- 当前评测只需要 remote 模型链路，不应因本地 `lmdeploy` 未安装阻塞框架 import。
- 自托管模型通过 OpenAI-compatible API 暴露，需要复用同一套 planner 调用路径。
- 本机 vLLM 使用 `--max-model-len=4096` 运行时，EmbodiedBench 默认 `max_tokens=4096` 会挤占全部上下文并导致 OpenAI-compatible 服务拒绝请求；512 足够承载动作 JSON 输出，也可按需通过环境变量调大。
- 本次全量评测 vLLM 使用 `--max-model-len=4096`，EB-ALFRED 观察到输入约 2049 token 时若仍请求 2048 输出 token 会被 OpenAI-compatible 服务拒绝；因此移除旧的 `2048` 下限，让 wrapper 能按需压低输出预算。这只影响模型一次回复的最大长度，不改动作解析、执行或评分逻辑。
- EB-ALFRED 部分 episode 的图像/历史上下文 prompt 可超过 3800 token，固定 `256` 输出预算仍可能刚好超过 4096 上下文；自适应重试用于避免这类样本被框架级 400 错误中断。全量 wrapper 同时把 EB invalid action 上限设为 `3`，避免连续格式错误把 prompt 继续膨胀到不可请求。
- 当前 Qwen3.5 vLLM 在 ALFRED 长 prompt + JSON schema 约束解码下可能生成大量空白并截断；该开关用于保留 prompt 内 JSON 约束和既有 `fix_json`/解析后处理，同时绕开服务端 schema 约束解码。
- 当前 served model name 是 `qwen35-vl`，不包含上游代码原先硬编码的 `Qwen2-VL` / `Qwen2.5-VL` / `Qwen3-VL` 片段；补齐该匹配后，后续 EB shard 会明确收到 JSON 输出格式约束。
- 不修改 prompt、response schema、解析逻辑或 evaluator 评分逻辑。

### Gemini 依赖可选化

涉及文件：

- `embodiedbench/planner/remote_model.py`
- `embodiedbench/planner/planner_utils.py`

修改内容：

- 将 `google.generativeai` 顶层导入改为可选导入；环境未安装 `google-generativeai` 时设置 `genai = None`，不阻塞 planner import。

原因和影响：

- 当前 EB-ALFRED / EB-Habitat 全量评测使用 Qwen3.5 vLLM 的 OpenAI-compatible API，不走 Gemini planner 路径。
- 最小评测镜像可能缺少 `google-generativeai`；直接安装该包会引入较重的 Google API 依赖并改动 protobuf 版本，可能影响 vLLM/平台环境。
- 该改动只避免未使用后端的顶层 import 失败；实际选择 Gemini 模型时仍需要运行环境安装对应依赖。

### AI2-THOR Unity Player 日志关闭

涉及文件：

- `embodiedbench/envs/eb_alfred/env/controller.py`

修改内容：

- EB-ALFRED 启动 AI2-THOR Unity Player 的 headless 命令从：
  ```text
  -batchmode -nographics
  ```
  调整为：
  ```text
  -batchmode -nographics -nolog
  ```

原因和影响：

- 本地并发评测时 Unity/AI2-THOR 会在工作目录生成 `*.excp` 异常日志副产物，污染项目根目录。
- `*.excp` 不是评测输入、模型输出、模拟执行记录或 scorer 结果；禁用 Unity Player logging 不改变 prompt、动作解析、环境交互或 task_success/task_progress/subgoal_reward 计算。
- 外层 wrapper 仍保留删除 `*.excp` 的兜底清理逻辑，用于防止 Unity 原生层异常在极端情况下绕过 `-nolog` 后继续落盘。
