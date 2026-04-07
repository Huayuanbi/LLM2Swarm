# Model Output Smoke Report (2026-04-07)

这份笔记记录了 2026 年 4 月 7 日对当前两层核心模型能力做的一次真实抽样：

- 云端角色分配：`GlobalOperator.assign_roles()`
- 机载初始任务图生成：`OnboardPlannerAgent.build_initial_task_graph()`

本次目标不是验证“任务已经足够聪明”，而是验证：

- 当前接口是否已经能承接真实模型输出
- 当前模型是否能给出“相对合理、结构化、不过分离谱”的结果
- 当前最主要的短板在哪一层

## 测试条件

- 日期：`2026-04-07`
- 机器位置：本地通过 SSH tunnel 访问远端 Ollama
- tunnel 状态：`localhost:11435 -> 10.130.138.37:11434`
- 可用模型：
  - `qwen3.5:9b`
  - `qwen3.5:4b`
- 抽样 mission：
  - `search the area for fire`

## 测试方法

这次没有直接依赖 Python `AsyncOpenAI` SDK 跑 end-to-end，因为在当前沙箱环境里：

- `curl -> localhost:11435` 可正常访问模型
- 但 Python SDK 路径会报 `Connection error`

因此本次采用的方法是：

1. 用仓库内当前代码生成真实 prompt payload
2. 直接通过 `curl` 调 `/v1/chat/completions`
3. 检查真实模型返回内容是否合理

这个方法验证的是：

- 当前 prompt
- 当前 schema
- 当前模型真实输出

而不是 mock 或手工构造回复。

## 一、Cloud role allocation (`qwen3.5:9b`)

### 使用接口

- 文件：[operators/global_operator.py](/Users/hyb/LocalProj/LLM2Swarm/operators/global_operator.py)
- 接口：

```python
await GlobalOperator.assign_roles(
    mission_description,
    initial_states=...,
    agent_profiles=...,
)
```

### 输入摘要

- mission：`search the area for fire`
- 3 架无人机，初始位置分别大致位于：
  - `drone_1`: `(-20, 10, 0)`
  - `drone_2`: `(15, -5, 0)`
  - `drone_3`: `(5, 25, 0)`
- 每架机都提供了 `AgentProfile`

### 输出概况

`qwen3.5:9b` 返回了合法 JSON，并且按 3 架机做了分工：

- `drone_1`：`sector_searcher_west`
- `drone_2`：`sector_searcher_south`
- `drone_3`：`sector_searcher_north`

每个 `RoleBrief` 都包含：

- `mission_role`
- `mission_intent`
- `responsibilities`
- `constraints`
- `coordination_contracts`
- `capability_requirements`
- `shared_context`
- `initial_hints`

而且 `shared_context` 里给出了分区边界，例如：

- west / south / north 三个 sector
- 每个 sector 对应的 bounds

### 这层表现好的地方

- 输出是结构化的，符合 `GlobalRolePlan` 方向
- 没有退回旧式 action list
- 会利用初始位置做相对合理的空间分工
- 会主动写 coordination contracts，体现“减少重复覆盖”的意图

### 这层目前的问题

模型会脑补一些并没有明确给出的能力、资源或规则，例如：

- `thermal_camera_payload`
- `fire_detection_algorithm`
- `access weather data`
- `report status every 30 seconds`
- `maintain minimum 50m separation`

这些内容本身不一定“荒谬”，但它们没有来自当前输入上下文，说明：

- 角色分配层的结构化能力已经够用
- 但 hallucination 仍然明显

### 结论

`qwen3.5:9b` 现在可以承担：

- 抽象 mission -> per-agent role brief

但还不能直接无约束信任，后续仍需要：

- 更严格 prompt
- 更强 validator
- 更明确的 capability/resource grounding

## 二、Onboard initial task graph (`qwen3.5:4b`)

### 使用接口

- 文件：[operators/onboard_planner.py](/Users/hyb/LocalProj/LLM2Swarm/operators/onboard_planner.py)
- 接口：

```python
await OnboardPlannerAgent.build_initial_task_graph(
    context: OnboardPlanningContext
)
```

### 输入摘要

- 使用一个通用 `RoleBrief`
- `mission_intent` 仍然是：`search the area for fire`
- 当前 agent：`drone_1`
- 当前状态：
  - on ground
  - battery high
- peers 也一起提供给了 onboard planner

### 输出概况

`qwen3.5:4b` 返回了合法 `TaskGraphSpec`，主要内容是：

1. `takeoff`
2. `search_pattern`

大意是：

- 先起飞到安全高度
- 然后围绕当前位置做一个小半径搜索 pattern

它给出的图结构是：

- entry node：`node_takeoff`
- edge：`node_takeoff -> node_search`
- `search_pattern` 的中心设在当前坐标附近
- 半径选择为较小的 bootstrap 半径

### 这层表现好的地方

- 输出是合法 JSON
- 图结构简单清晰
- 没有调用不存在的 primitive
- 没有无视当前 agent 的能力边界
- 很符合“先给一个安全 bootstrap graph”这个定位

### 这层目前的问题

- 规划明显偏保守
- 更像“安全起步图”，不是强任务规划
- 对 mission 的理解还比较浅，只给出本地起步而不是更完整的局部策略

但这并不是坏事。对于当前阶段来说，这说明：

- `RoleBrief -> TaskGraphSpec` 这条接口已经能承接真实模型输出
- 即使模型能力一般，也能先生成“不离谱”的初始图

### 结论

`qwen3.5:4b` 现在可以承担：

- 机载 bootstrap task graph 生成

它还不够强，但足以支撑当前框架验证。

## 三、Runtime multimodal VLM (`qwen3.5:4b`)

### 使用接口

- 文件：[operators/vlm_agent.py](/Users/hyb/LocalProj/LLM2Swarm/operators/vlm_agent.py)
- 接口：

```python
await VLMAgent.decide(...)
```

### 输入摘要

- 使用 mock controller 生成当前相机图像
- 同时提供：
  - telemetry
  - peer states
  - available primitives
  - available capabilities

### 结果

这次没有拿到一个及时返回的、可直接分析的最终结果。

表现为：

- 多模态请求长时间无返回
- 明显超过当前运行时默认的 `VLM_TIMEOUT=20.0s`

### 这意味着什么

这不是说当前 VLM 一定“语义不合理”，而是说明：

- 当前 `qwen3.5:4b` 在这条多模态 runtime 路径上，实时性明显不足
- 在真实运行中，它大概率会因为超时而触发 fallback
- 当前这层的主要瓶颈是性能 / 时延，而不是 schema 接不住

### 结论

`qwen3.5:4b` 当前不适合直接作为高频、强实时的 runtime VLM 决策器。

如果后续要提升这一层，优先方向应该是：

- 更轻量的视觉输入策略
- 更快的模型
- 更强的本地 agent / 分步工具链
- 或者把 runtime VLM 的职责进一步收窄

## 总结判断

截至 `2026-04-07`，当前框架对真实模型输出的承接情况可以总结为：

- Cloud role layer：
  - `可用`
  - 结构化能力基本够
  - 需要继续压 hallucination

- Onboard bootstrap planning：
  - `可用`
  - 适合作为初始图生成器
  - 当前偏保守，但方向正确

- Runtime multimodal VLM：
  - `接口已通`
  - `实时性不足`
  - 当前仍更像一个可接入组件，而不是成熟运行组件

## 当前建议

后续如果继续提升模型能力，建议优先顺序：

1. 先优化 `GlobalOperator.assign_roles()` 的 grounding 和 hallucination
2. 再优化 `OnboardPlannerAgent` 的 bootstrap graph 质量
3. 最后集中处理 `VLMAgent.decide()` 的多模态实时性

## 备注

本次报告只验证了“当前 prompt + 当前 schema + 当前模型输出”的基本合理性。

它不代表：

- 模型已经足够强
- 真实多机场景已经完全稳定
- 当前 prompt 已经收敛

它的意义在于确认：

- 当前接口设计方向是可行的
- 框架已经能承接真实模型输出
- 接下来可以把主要精力放在模型能力提升和 prompt 迭代上
