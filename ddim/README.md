# DDIM 少步采样实验

`core/diffusion_sampling.py` 是 DDIM 更新和时间步网格；`core/ddim_ddpm_metrics.py` 是轨迹比较指标；`core/entry.py` 是各入口共享的观测加载、模型构建与指标聚合。采样调用仍通过基线教师类接入，不训练额外 student。

入口统一为 `cli.py`：`benchmark`（时延）、`compare`（DDPM/DDIM 轨迹分布对比）、`sweep`（sampler/step/RTC 矩阵）；`scripts/capture_ddim_ddpm_obs.sh` 负责采集固定观测。五步常用网格为 `[9,7,4,2,0]`；RTC 是另一个实验变量。

从根目录用 `python run.py ddim-benchmark --help`、`python run.py ddim-compare --help`、`python run.py ddim-sweep --help`。基准工具不是 smoke：合成输入只能测执行/计算，真实观测重放也不能替代闭环成功率/碰撞率。
