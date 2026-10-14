# CHANGELOG.md

本文档记录本项目对官方 ET-Plan-Bench 框架的本地必要修改。范围仅限 Embodied Arena Task Planning 子榜单中的 `ET-Plan_Bench`。

维护原则：

- 优先通过项目根目录 wrapper、环境变量和外部依赖目录适配运行环境。
- 只有框架内部路径、模拟器初始化或 scorer 兼容性无法在外部处理时，才修改本目录源码。
- 后续修改 `ET-Plan-Bench/` 内源码、软链接或运行配置时，必须同步更新本文档。

## 2026-07-15 single-object 房间候选空返回兼容

涉及文件：

- `evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/utils/robot.py`

修改内容：

- `robot_exploration()`、`robot_exploration_constraint()` 和 `robot_exploration_abstract()` 在 `LLM_find_rooms()` 返回 `None` 时走未找到目标的失败路径，不再直接迭代 `None`。
- 若模型返回单个字符串候选，则规范为单元素列表后继续沿用官方遍历逻辑。

原因和影响：

- 官方 evaluator 在模型将房间候选回答为 `None` 时会触发 `TypeError: 'NoneType' object is not iterable`，使本可按失败轨迹计分的 task 变成 retryable task error。
- 修改只处理异常房间候选的容错分支；有效候选的房间遍历、动作执行、轨迹保存和评分逻辑保持不变。

## 2026-07-14 multi-object 空房间候选兼容

涉及文件：

- `evaluation_generation/navigation_manipulation_multiple_objects/inference_exploration_inside_both.py`
- `evaluation_generation/navigation_manipulation_multiple_objects/inference_exploration_temp.py`

修改内容：

- 候选房间名称仍存在、但对应房间实例已被前序探索耗尽时，跳过该候选，不再直接访问空的距离排序结果。

原因和影响：

- 官方 evaluator 在该分支会触发 `IndexError`，使本可按失败轨迹计分的 task 变成 retryable task error。
- 修改与同目录其他 evaluator 的既有空列表处理一致，不改变有效房间候选的动作、轨迹或评分语义。

## 2026-07-02 运行临时文件收敛

涉及文件：

- `simulator/virtualhome/virtualhome/simulation/unity_simulator/communication.py`

修改内容：

- `UnityCommunication` 新增支持环境变量 `ET_PLAN_RUNTIME_DIR`。
- 设置该变量后，VirtualHome 生成的 `port_*.txt` 和 `Player_*.log` 写入该目录；未设置时仍保持官方默认行为，写入当前工作目录。
- 设置该变量后，Unity 子进程的工作目录也切换到该目录，避免 Unity 运行时生成的 UUID 临时目录散落在项目根目录。
- `ET_PLAN_DISABLE_PORT_FILE` 默认关闭 `port_*.txt` 标记文件写入；如需恢复旧行为，可显式设置 `ET_PLAN_DISABLE_PORT_FILE=0`。

原因和影响：

- 全量评测会并发启动大量 Unity 实例，官方默认会把端口日志和 Unity 日志散落在项目根目录。
- 该修改只改变临时日志的保存位置和端口标记文件写入行为，不影响 simulator 端口、动作执行、模型输入输出、后处理或评分逻辑。

## 2026-07-06 single-object VirtualHome headless 初始化补齐

涉及文件：

- `evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/envs/env.py`
- `evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/utils/log.py`

修改内容：

- single-object navigation-manipulation 环境补齐与 navigation / multi-object 环境一致的 headless 初始化开关：
  ```text
  ET_PLAN_SKIP_START_UNITY
  ET_PLAN_NO_GRAPHICS
  ET_PLAN_X_DISPLAY
  ET_PLAN_TIMEOUT_WAIT
  ```
- `VirtualHomeEnv.close()` 在存在本地启动的 Unity 进程时主动 kill，减少残留进程。
- `get_logger()` 在从项目根 wrapper 调用、主入口路径不包含 `EmbodiedAI` 仓库名时，退回使用文件 basename 生成日志路径，避免日志初始化 `IndexError`。

原因和影响：

- 官方 single-object 环境原始实现强制调用 `start_unity()` 并固定 `x_display="0"`，与本机 AMD/headless 运行环境不适配，也与本项目此前已完成的 navigation / multi-object no-graphics 初始化不一致。
- 官方 single-object 日志工具假设入口脚本位于 `EmbodiedAI` 目录内；通过项目根 `scripts/` wrapper 调用官方函数时，该假设不成立。
- 这些修改只影响 Unity 启动/清理方式和日志文件路径，不改变 action DSL、observation、TaskEnv 记忆图、任务条件解析、动作执行或成功判定逻辑。

## 2026-07-06 navigation large-object id 初始化修复

涉及文件：

- `evaluation_generation/navigation_tasks/gpt4_navigation.py`

修改内容：

- 在 `robot_exploration_without_constraint()` 处理模型返回的候选 large object 时，补齐：
  ```python
  large_object_ids = [key for key, name in surround_visible_objects_dict.items() if name == large_object]
  ```

原因和影响：

- 官方同目录的 `gpt4_navigation_rationale_CoT.py` 和 `gpt4_navigation_layout_map.py` 均有这一行，当前正式使用的 `gpt4_navigation.py` 漏掉后，会在模型返回可见物体名称时触发 `UnboundLocalError: local variable 'large_object_ids' referenced before assignment`。
- 该修改只补齐实例 id 列表初始化，使原本后续的官方 `distance_from_agent()`、`[walk]`、`[open]`、可见图刷新和成功判断逻辑能够继续执行；不改变 prompt、任务条件、动作模板或评分规则。

## 2026-07-01 AMD/ROCm 本机适配

### VirtualHome simulator 运行时链接

修改内容：

```text
simulator -> ${TP_ET_SIMULATOR_ROOT:-<仓库父目录>/tp-deps/ET-Plan-Bench/simulator}
```

原因和影响：

- VirtualHome v2.3.0 simulator 是大体积二进制依赖，不放入项目源码目录。
- ET-Plan-Bench 内部环境初始化和项目 wrapper 均依赖仓库根目录下的 `simulator` 路径。
- 链接不再提交到 Git；统一入口按当前同事的仓库位置或 `.env` 中的 `TP_ET_SIMULATOR_ROOT` 自动创建。
- `scripts/prepare_external_runtime.py` 会幂等应用 VirtualHome 并发运行所需的 runtime-dir、日志和进程隔离补丁，避免依赖某个用户预先修改好的外部 simulator。

### VirtualHome headless 初始化

涉及文件：

- `evaluation_generation/navigation_tasks/envs/env.py`
- `evaluation_generation/navigation_manipulation_multiple_objects/envs/env.py`

修改内容：

- Unity executable 路径改为基于源码位置构造绝对路径，避免依赖运行时当前目录。
- 默认使用 `no_graphics=True`，并支持以下环境变量：
  ```text
  ET_PLAN_SKIP_START_UNITY
  ET_PLAN_NO_GRAPHICS
  ET_PLAN_X_DISPLAY
  ET_PLAN_TIMEOUT_WAIT
  ```
- `close()` 在存在本地启动的 Unity 进程时主动 kill，减少残留进程。

原因和影响：

- 本机 AMD/headless 环境需要稳定连接 VirtualHome simulator，不能依赖固定相对路径或图形显示。
- 这些改动只影响模拟器启动和清理方式，不修改动作执行、任务条件或成功判定逻辑。

### task condition 兼容

涉及文件：

- `evaluation_generation/navigation_tasks/tasks/task_checker.py`
- `evaluation_generation/navigation_manipulation_multiple_objects/tasks/task_checker.py`

修改内容：

- task condition 解析时将主体名 `robot` 规范化为 VirtualHome 场景图中的 `character`。

原因和影响：

- 官方 benchmark 的 completion criterion 使用 `robot`，而 VirtualHome 执行图和 checker 合法类名使用 `character`。
- 该修改只做主体别名对齐，使 scorer 能正确匹配同一个 agent；不放宽物体关系、不新增成功条件、不改变 `TaskChecker.is_success()` 的图关系匹配逻辑。

### multi-object task 条件预解析

涉及文件：

- `evaluation_generation/navigation_manipulation_multiple_objects/envs/env.py`

修改内容：

- 加载 multi-object task 时预解析 `task_conditions`。

原因和影响：

- multi-object / temporal smoke 需要复用 `TaskChecker.is_success()`，而该路径要求任务信息中已存在 `task_conditions`。
- 预解析结果来自原始 `task_completion_criterion`，不改变任务目标或评分规则。
