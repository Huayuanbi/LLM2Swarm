# LLM2Swarm

一个由大语言模型驱动的云边协同多无人机系统原型。

这个项目的核心目标是把自然语言任务分解成多架无人机的动作序列，并在执行过程中持续结合机载视觉进行重规划。

当前仓库同时支持三种运行形态：

- `mock` 纯 Python 仿真，适合离线测试和主流程验证
- `Webots` 单机 demo，已验证可以起飞并完成 `camera -> VLM -> action` 闭环
- `Webots` 多机 demo，采用“云端先统一生成任务，再平等下发给各机”的初始化方式，并已实测跑通 `takeoff -> go_to_waypoint -> search_pattern`

## 核心思路

系统分成两层：

- 云端全局规划：`GlobalOperator` 接收自然语言任务，输出 `drone_id -> action list`
- 边端局部重规划：每架无人机执行动作时，周期性把相机图像、位姿、速度和队友状态发给 `VLMAgent`，让 VLM 判断是继续还是修改任务

其中每架无人机都有自己的独立控制循环：

1. 从任务队列取出当前动作
2. 开始执行动作
3. 同时触发一次 VLM 观察与决策
4. 如果 VLM 返回 `continue`，继续执行
5. 如果 VLM 返回 `modify`，取消当前动作并把新动作插回队列

VLM 调用和动作执行是并发的，无人机不会停下来等待模型返回。

## 当前状态

已经可用的部分：

- `mock` 后端完整可跑
- 单机 Webots controller 已接通
- Webots 相机图像已能传给 VLM
- `VLMAgent` 已能通过 SSH tunnel 访问远端 Ollama / qwen
- Webots 单机 demo 已实测能正常启动并完成起飞
- Webots 多机 demo 已实测能完成起飞、到达初始航点并进入搜索模式
- 多机 world 已为每架无人机添加彩色 marker，便于在场景中快速识别

正在持续完善的部分：

- 多机协同任务效果验证
- 更稳定的 Webots 世界与任务配置
- 更强的多机避碰与区域约束

## 整体架构

```text
Natural-language mission
        |
        v
Cloud planner (GlobalOperator)
        |
        v
Per-drone action lists
        |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
   drone_1 loop         drone_2 loop         drone_3 loop
        |                    |                    |
        | execute action     | execute action     | execute action
        |       +            |       +            |       +
        |   VLM tick         |   VLM tick         |   VLM tick
        |                    |                    |
        +---------- shared swarm state -----------+
                           |
                           v
                    peer states / observations
```

在 Webots 中，飞控和 agent 也是分层的：

- 高频飞控循环：常驻在 `WebotsController` 内部，负责 `robot.step()`、传感器读取、姿态控制和电机输出
- 低频 agent 循环：常驻在 `DroneLifecycle`，负责动作推进和 VLM tick

## 目录说明

```text
LLM2Swarm/
├── main.py                               # mock / headless 入口
├── config.py                             # 全局配置
├── requirements.txt
├── controllers/
│   ├── base_controller.py                # 控制器抽象接口
│   ├── mock_controller.py                # 纯 Python mock 控制器
│   ├── webots_controller.py              # Webots 飞控状态机 + 常驻 step 循环
│   ├── llm2swarm_single/
│   │   └── llm2swarm_single.py           # 单机 Webots demo controller
│   └── llm2swarm_multi/
│       └── llm2swarm_multi.py            # 多机 Webots demo controller
├── memory/
│   ├── pool.py                           # 进程内共享状态池
│   └── sqlite_pool.py                    # 多进程 Webots 共享状态池
├── operators/
│   ├── global_operator.py                # 云端全局规划
│   ├── vlm_agent.py                      # 边端视觉决策
│   └── drone_lifecycle.py                # 单机生命周期控制循环
├── models/
│   └── schemas.py                        # Pydantic schema
├── utils/
│   ├── image_utils.py                    # 图像编码
│   └── visualizer.py                     # mock 运行时可视化
├── worlds/
│   ├── mavic_2_pro_llm_demo.wbt          # 单机 Webots demo
│   └── mavic_2_pro_swarm_demo.wbt        # 多机 Webots demo
├── scripts/
│   ├── start_tunnel.sh                   # SSH tunnel 启动脚本
│   ├── run_webots_single_demo.sh         # 单机 Webots 启动脚本
│   ├── prepare_webots_swarm_plan.py      # 多机云端初始化脚本
│   ├── run_webots_swarm_demo.sh          # 多机 Webots 启动脚本
│   ├── benchmark_thinking.py
│   └── test_concurrent_vlm.py
└── tests/
    ├── test_phase1.py
    ├── test_phase2.py
    └── test_phase3.py
```

## 环境要求

- macOS 或 Linux
- Python 3.11+
- conda 环境 `llm2swarm`
- 远端 Ollama 服务
- 可访问的 qwen 模型
- SSH 到远端 Ollama 机器的权限

推荐安装步骤：

```bash
git clone https://github.com/Huayuanbi/LLM2Swarm.git
cd LLM2Swarm

conda create -n llm2swarm python=3.11 -y
conda activate llm2swarm
pip install -r requirements.txt
```

## 配置

先复制环境变量模板：

```bash
cp .env.example .env
```

典型 `.env` 配置如下：

```env
OPENAI_API_KEY=sk-...

SIMULATOR_BACKEND=mock
LOG_LEVEL=INFO

# Edge VLM
EDGE_VLM_BASE_URL=http://localhost:11435/v1

# Global planner
GLOBAL_LLM_BASE_URL=http://localhost:11435/v1
GLOBAL_LLM_API_KEY=ollama
GLOBAL_LLM_MODEL=qwen3.5:9b

NO_PROXY=localhost,127.0.0.1
no_proxy=localhost,127.0.0.1
```

说明：

- `EDGE_VLM_BASE_URL` 是边端 VLM 的入口，通常通过 SSH tunnel 映射到本地
- `GLOBAL_LLM_*` 控制初始任务规划器
- 若远端 GPU 容量较紧，建议把 `GLOBAL_LLM_MODEL` 调成 `qwen3.5:4b`，联调时会更稳
- 若不想让云端规划参与，可以清空 `GLOBAL_LLM_BASE_URL` 和 `OPENAI_API_KEY`

## SSH Tunnel

先启动通往远端 Ollama 的 tunnel：

```bash
./scripts/start_tunnel.sh
```

查看状态：

```bash
./scripts/start_tunnel.sh --status
```

停止：

```bash
./scripts/start_tunnel.sh --kill
```

如果你的网络环境使用了本地代理，确保 `localhost` 被 `NO_PROXY` 排除。

## 运行方式

### 1. Mock 模式

这是最适合快速验证主逻辑的入口：

```bash
conda activate llm2swarm
python main.py
```

带可视化：

```bash
python main.py --visualize
```

自定义任务：

```bash
python main.py --mission "Drone 1 patrol sector A, Drone 2 standby at base"
```

### 2. 单机 Webots Demo

这是当前最推荐的 Webots 调试入口，已经打通了 `camera -> VLM -> action` 闭环。

启动方法：

```bash
./scripts/start_tunnel.sh
./scripts/run_webots_single_demo.sh
```

它会打开：

- `worlds/mavic_2_pro_llm_demo.wbt`

其中使用的 Webots controller 是：

- `controllers/llm2swarm_single/llm2swarm_single.py`

启动逻辑：

1. 先尝试用 `GlobalOperator` 生成初始 action list
2. 若失败则退回内置任务
3. 启动 `DroneLifecycle`
4. 在动作执行过程中周期性调用 `VLMAgent`
5. VLM 根据 camera 图像返回 `continue` 或 `modify`

### 3. 多机 Webots Demo

多机版本现在采用“云端先初始化，全局平等下发”的方式。

启动方法：

```bash
./scripts/start_tunnel.sh
./scripts/run_webots_swarm_demo.sh
```

它会先执行：

```bash
python scripts/prepare_webots_swarm_plan.py
```

也就是：

- 先重置共享 SQLite 状态库
- 云端统一生成完整 `drone_id -> action list`
- 将全局计划写入共享存储

然后再打开：

- `worlds/mavic_2_pro_swarm_demo.wbt`

这个 world 中每架 `Mavic2Pro` 都运行：

- `controllers/llm2swarm_multi/llm2swarm_multi.py`

为了方便观察，多机 world 还给三架无人机分别挂了醒目的彩色 marker：

- `drone_1`：红色
- `drone_2`：绿色
- `drone_3`：蓝色

每架无人机启动后只做这几件事：

1. 从共享 SQLite 读取属于自己的初始任务
2. 启动自己的 `WebotsController`
3. 启动自己的 `DroneLifecycle`
4. 周期性执行自己的 VLM tick
5. 将自身状态写回 SQLite，供其他无人机读取

共享状态文件默认是：

- `/tmp/llm2swarm_webots_swarm.sqlite3`

## Webots 控制逻辑

`WebotsController` 当前不是简单的“发命令即返回”，而是一个常驻飞控状态机：

- 内部维护后台线程
- 独占 `robot.step()`
- 每个 step 读取 `camera / gps / imu / gyro`
- 将最近一帧 camera 缓存为 Base64 JPEG
- 根据当前 mode 计算四个电机转速

高层 action 只做“设定目标 + 等待完成”，而不是直接控制电机。

当前已实现的动作：

- `takeoff`
- `go_to_waypoint`
- `hover`
- `search_pattern`
- `land`

其中：

- `search_pattern` 是持续动作，不会自己结束
- 默认会沿内部圆周航点持续循环
- 需要 VLM 返回 `modify` 才会切换到新任务

## VLM 输入与输出

每个 VLM tick 会发送：

- 当前机载 camera 图像
- 当前位姿和速度
- 当前任务
- 队友状态摘要

VLM 返回两种 JSON 之一：

```json
{ "decision": "continue" }
```

或

```json
{
  "decision": "modify",
  "new_task": "avoid obstacle ahead",
  "new_action": {
    "action": "go_to_waypoint",
    "params": { "x": 5, "y": 5, "z": 10, "velocity": 3 }
  },
  "memory_update": "obstacle detected near windmill"
}
```

如果 VLM 超时或输出非法 JSON，系统会自动回退为 `continue`。

## 测试

项目自带三阶段测试，主要覆盖 mock 路径：

```bash
conda activate llm2swarm

python tests/test_phase1.py
python tests/test_phase2.py
python tests/test_phase3.py
```

目前测试特点：

- `Phase 1` 纯离线，稳定可跑
- `Phase 2` 的在线规划测试依赖可访问的模型接口
- `Phase 3` 的 live VLM 依赖 SSH tunnel
- Webots 单机与多机 demo 需要在真实 Webots GUI 中观察，不包含在这三组自动化测试里

## 当前推荐的调试顺序

如果你是第一次接触这个仓库，建议按这个顺序跑：

1. `python tests/test_phase1.py`
2. `./scripts/start_tunnel.sh --status`
3. `python main.py`
4. `./scripts/run_webots_single_demo.sh`
5. `./scripts/run_webots_swarm_demo.sh`

这样最容易定位问题到底是在：

- Python 环境
- 模型连通性
- 生命周期逻辑
- 单机 Webots 飞控
- 还是多机协同

## 设计说明与限制

### 为什么先做 direct controller

当前 Webots demo 先采用 direct controller，是因为它更适合把以下链路快速打通：

- 飞控
- 相机
- VLM 调用
- 动作执行

单机路径已经验证成功后，再回到更复杂的 extern-controller 组织方式会更稳。

### 多机为什么要用 SQLite

在 Webots 多机 direct controller 模式下，每架无人机都是独立 controller 进程，所以不能再用进程内的 `SharedMemoryPool`。因此多机 demo 使用：

- `memory/sqlite_pool.py`

来共享：

- 位置
- 速度
- 状态
- 观察
- 初始任务

## 后续建议

如果你准备继续推进这个项目，最值得优先做的是：

- 调整 global plan prompt，让初始 waypoint 更符合场景
- 优化 `search_pattern` 的退出条件
- 为多机 Webots 增加更明确的碰撞规避动作
- 给多机 demo 加日志面板或状态可视化
