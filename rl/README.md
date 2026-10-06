# 学习型导航执行器

`src/` 是实际实现，支持 direct 和 residual 两种模式。`direct` 直接输出 `(v, omega)`，以替代 MPC 执行为目标；`residual` 仍依赖 MPC 名义命令，作为对照保留。

- 输入：当前 RGB-D、机器人状态、对齐后的参考轨迹；融合模块和 GRU 建模控制时序。
- 训练：`cli.py collect` / `cli.py collect-all` 采集 BC 教师数据，`cli.py train-bc` 初始化，`cli.py train-tracker` 做 PPO，`cli.py train-full` 调度多场景；residual 对照为 `cli.py train-reactive`。
- 编码器：`src/encoder.py` 包含工作区已有的可切换视觉编码器路径；外部依赖集中在根目录 `third_party/`，未复制权重。
- `src/runtime.py` 管理异步特征与执行；评估集成实际实现在 `bridge/planner_executor_bridge.py`。
- `src/entry.py` 是各入口共享的场景/调度/基准辅助；配置在 `config/`，acados 环境封装在 `scripts/*.sh`。

## 网络

- **视觉编码**（`src/encoder.py`）：可切换 RGB-D 后端（DA-V2 / YOLO26-Depth），输出 RGB/depth token。
- **融合 + 策略**（`src/policy.py`）：`RGBDFusion` 把 RGB/depth token 分别投影，拼接后与可学习 query 做多头注意力 + FFN，融合机器人状态、参考轨迹与 goal；`ReactiveActorCritic` 用 GRU 建模控制时序，actor 输出 `(v, omega)`，critic 输出价值。

## 奖励（direct PPO，`config/reactive_rgbd_direct_g1.yaml`）

加权和，主项（括号为权重）：

- 前进/跟踪：`progress_s`(1.0)、`tracking_recovery`(2.0)、`tracking_corridor`(1.0)
- 安全：`clearance`(0.75，避障净空)、`ttc`(0.4，碰撞时间)、`smoothness`(0.05)
- 目标逼近：`goal_approach_progress`(2.0)/`speed`(0.5)/`capture`(2.0)/`hold`(2.0)/`stop`(0.2)/`brake`(0.25)
- 惩罚/终止：`stall`(0.1)、`time`(0.01)、`contact`(2.0)、`collision`(10.0)、`fall`(10.0)、`oob`(10.0)、`success`(10.0)

## 损失

- **BC 初始化**（`train-bc`）：actor 输出与教师命令的 MSE。
- **PPO**（`train-tracker`/`train-full`）：RSL-RL 的 surrogate + value + entropy。

根目录入口：`python run.py rl-collect --help`、`python run.py rl-bc --help`、`python run.py rl-train --help`、`python run.py rl-multiscene --help`、`python run.py rl-residual --help`、`python run.py rl-bench --help`。默认工作目录为 `baselines/x-navdp/`。不要将学习型导航速度执行表述为重新训练机器人步态或关节控制器。
