# TODO

这份清单只记录“框架层”的后续工作，不记录某个具体场景的写死策略。

## P0: 接口稳定性

- 为 `GlobalOperator.assign_roles()` 增加更多 contract tests
- 为 `OnboardPlannerAgent.build_initial_task_graph()` 增加更多 contract tests
- 为 `VLMAgent.decide()` 增加更强的结构化输出 contract tests
- 明确 `RoleBrief -> TaskGraphSpec -> VLMDecision` 的版本兼容策略

## P0: 模型升级接口

目标：后续可以把当前 `llm / vlm` 替换成更强模型或 agent framework，而不重写 runtime。

需要保持稳定的接口：

- 云端角色分配：
  - 输入：`mission + initial_states + agent_profiles`
  - 输出：`GlobalRolePlan`
  - 文件：
    - [operators/global_operator.py](/Users/hyb/LocalProj/LLM2Swarm/operators/global_operator.py)

- 机载初始任务图生成：
  - 输入：`OnboardPlanningContext`
  - 输出：`OnboardPlanningResponse`
  - 文件：
    - [operators/onboard_planner.py](/Users/hyb/LocalProj/LLM2Swarm/operators/onboard_planner.py)

- 运行时 VLM / 本地智能：
  - 输入：当前观测 + 当前共享上下文 + 当前能力约束
  - 输出：`VLMDecision`
  - 文件：
    - [operators/vlm_agent.py](/Users/hyb/LocalProj/LLM2Swarm/operators/vlm_agent.py)

建议后续工作：

- 把 `GlobalOperator` 后端替换成 planner agent
- 把 `OnboardPlannerAgent` 升级成多步 agent
- 把 `VLMAgent` 升级成支持工具调用的本地 agent
- 给三层 planner 都增加 prompt/version metadata
- 精简 `OnboardPlannerAgent` prompt，减少重复上下文，提供更适合弱模型的 bootstrap-only 规划模式

## P0: 新增动作的标准入口

新增动作时，标准修改点如下：

1. 在 [primitives/registry.py](/Users/hyb/LocalProj/LLM2Swarm/primitives/registry.py) 注册新的 `PrimitiveSpec`
2. 在对应 backend 里实现 handler
   - [controllers/mock_controller.py](/Users/hyb/LocalProj/LLM2Swarm/controllers/mock_controller.py)
   - [controllers/webots_controller.py](/Users/hyb/LocalProj/LLM2Swarm/controllers/webots_controller.py)
3. 如果动作代表新的能力/资源，更新 `capability_tags / resource_tags`
4. 增加测试

建议后续工作：

- 支持从外部插件/配置动态加载 primitives
- 支持按 `agent_kind` 自动加载不同 primitive 组合
- 为 primitive 增加更严格的参数 schema 校验

## P1: AgentProfile 完善

目标：让框架更自然地支持无人机、地面机器人、载荷平台等异构 agent。

建议后续工作：

- 将 `agent_kind`、`available_resources`、`hard_constraints` 进一步结构化
- 允许从配置文件或数据库加载 agent profile
- 允许 Webots world / 真实设备初始化时自动生成 agent profile
- 增加不同 agent 类型的 sample profiles

相关文件：

- [models/schemas.py](/Users/hyb/LocalProj/LLM2Swarm/models/schemas.py)
- [primitives/registry.py](/Users/hyb/LocalProj/LLM2Swarm/primitives/registry.py)
- [scripts/prepare_webots_swarm_plan.py](/Users/hyb/LocalProj/LLM2Swarm/scripts/prepare_webots_swarm_plan.py)

## P1: TaskGraph runtime 增强

runtime 目前仍偏“线性 primitive 执行”。

建议后续工作：

- 真正支持 `decision` 节点
- 真正支持 `wait_event` 节点
- 真正支持 `claim` 节点语义
- 支持分支、回退、失败转移
- 支持 graph-level timeout / retry / abort policy
- 支持 graph patch merge / replace 策略

相关文件：

- [operators/local_planner.py](/Users/hyb/LocalProj/LLM2Swarm/operators/local_planner.py)
- [models/schemas.py](/Users/hyb/LocalProj/LLM2Swarm/models/schemas.py)

## P1: VLM / event / patch 语义增强

目标：让模型产生的不是“拍脑袋改动作”，而是更稳定的结构化 runtime 信号。

建议后续工作：

- 明确 event taxonomy
- 明确 patch taxonomy
- 区分 `observation`、`event`、`claim request`、`graph patch`
- 增加 runtime 对非法 patch 的诊断信息
- 增加 `memory_update` 的结构化版本

相关文件：

- [operators/vlm_agent.py](/Users/hyb/LocalProj/LLM2Swarm/operators/vlm_agent.py)
- [models/schemas.py](/Users/hyb/LocalProj/LLM2Swarm/models/schemas.py)
- [operators/drone_lifecycle.py](/Users/hyb/LocalProj/LLM2Swarm/operators/drone_lifecycle.py)

## P1: 多机共享协同能力

目标：体现真正的群体智能，而不是多架单机并行。

建议后续工作：

- 继续完善 claim / lease 生命周期
- 为不同任务类型设计更清晰的共享事件模式
- 支持任务接管、目标交接、观察确认、区域声明
- 增加群体冲突检测
- 增加多 agent 之间的轻量协商协议

相关文件：

- [memory/pool.py](/Users/hyb/LocalProj/LLM2Swarm/memory/pool.py)
- [memory/sqlite_pool.py](/Users/hyb/LocalProj/LLM2Swarm/memory/sqlite_pool.py)
- [operators/drone_lifecycle.py](/Users/hyb/LocalProj/LLM2Swarm/operators/drone_lifecycle.py)

## P2: Webots / simulator 侧

建议后续工作：

- 为 Webots 增加更多 primitive
- 将能力约束映射到真实 simulator 限制
- 增加最小端到端 smoke test
- 增加更容易观察的调试可视化
- 将单机、多机 demo 的 sample brief/sample graph 继续去样例化

相关文件：

- [controllers/webots_controller.py](/Users/hyb/LocalProj/LLM2Swarm/controllers/webots_controller.py)
- [worlds/mavic_2_pro_llm_demo.wbt](/Users/hyb/LocalProj/LLM2Swarm/worlds/mavic_2_pro_llm_demo.wbt)
- [worlds/mavic_2_pro_swarm_demo.wbt](/Users/hyb/LocalProj/LLM2Swarm/worlds/mavic_2_pro_swarm_demo.wbt)

## P2: 测试补强

建议后续工作：

- 增加 `OnboardPlannerAgent` fallback path 的 contract test
- 增加 `VLMDecision.task_graph_patch` 的 contract test
- 增加更严格的 capability mismatch tests
- 增加 registry 扩展测试
- 增加 Webots 启动 smoke test

相关文件：

- [tests/test_phase1.py](/Users/hyb/LocalProj/LLM2Swarm/tests/test_phase1.py)
- [tests/test_phase2.py](/Users/hyb/LocalProj/LLM2Swarm/tests/test_phase2.py)
- [tests/test_phase3.py](/Users/hyb/LocalProj/LLM2Swarm/tests/test_phase3.py)

## 原则

- 不把某个具体场景策略写死在框架里
- 不把“模型聪明程度”当成 runtime 正确性的前提
- 优先稳定接口，再提升模型能力
- 能力约束、资源约束、图校验、共享记忆这些基础设施优先于场景特化逻辑
