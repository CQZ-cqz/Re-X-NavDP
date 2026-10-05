# 学习型导航执行器

`core/` 是实际实现，支持 direct 和 residual 两种模式。`direct` 直接输出 `(v, omega)`，以替代 MPC 执行为目标；`residual` 仍依赖 MPC 名义命令，作为对照保留。

- 输入：当前 RGB-D、机器人状态、对齐后的参考轨迹；融合模块和 GRU 建模控制时序。
- 训练：`cli.py collect` / `cli.py collect-all` 采集 BC 教师数据，`cli.py train-bc` 初始化，`cli.py train-tracker` 做 PPO，`cli.py train-full` 调度多场景；residual 对照为 `cli.py train-reactive`。
- 编码器：`core/encoder.py` 包含工作区已有的可切换视觉编码器路径；外部依赖集中在根目录 `third_party/`，未复制权重。
- `core/runtime.py` 管理异步特征与执行；评估集成实际实现在 `bridge/planner_executor_bridge.py`。
- `core/entry.py` 是各入口共享的场景/调度/基准辅助；配置在 `config/`，acados 环境封装在 `scripts/*.sh`。

根目录入口：`python run.py rl-collect --help`、`python run.py rl-bc --help`、`python run.py rl-train --help`、`python run.py rl-multiscene --help`、`python run.py rl-residual --help`、`python run.py rl-bench --help`。默认工作目录为 `x-navdp/`。不要将学习型导航速度执行表述为重新训练机器人步态或关节控制器。
