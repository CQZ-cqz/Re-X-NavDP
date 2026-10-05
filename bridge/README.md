# 基线与二次开发的共享边界

- `teacher_adapter.py`：加载原教师推理权重，冻结参数；FM 标注/训练的 teacher loader 已调用。
- `trajectory_adapter.py`：统一原始增量到执行轨迹、Q 输入轨迹的两条后处理；FM `generate_and_rank` 已调用。
- `q_evaluator.py`：部署时双 Q 的均值评价；不代替 RL 训练使用的 min-Q 语义。
- `planner_executor_bridge.py`：从原 reactive eval_bridge 迁入的实际评估接线。
- `recovery/`：公共脱困状态和选择支持。

不把 `bridge` 当作复制教师模型的另一个目录。原教师网络仍在 `x-navdp/eval/src/`，现有服务层在迁移初版保留；后续拆分须保持数据格式和控制语义。
