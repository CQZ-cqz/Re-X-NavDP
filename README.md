# Re-X-NavDP

基于 [X-NavDP](https://github.com/InternRobotics/NavDP/tree/master/baselines/x-navdp) 的复现与二次开发。

| 模块 | 内容 | 主目录 |
| --- | --- | --- |
| RL  | 实时感知导航执行器，输出 `(v, omega)` 替代 MPC；MPC 教师 BC 初始化 + direct PPO | [rl](rl/README.md) |
| FM 蒸馏 | 小型条件 Flow Matching 蒸馏教师扩散解码器；采集/标注/训练/闭环评估 | [FM_distillation](FM_distillation/README.md) |
| DDIM | 原权重下 DDPM/DDIM 少步采样与时延对照 | [ddim](ddim/README.md) |

## 环境与资产

- **Python**：navrl conda 环境；系统 `python` 缺少依赖。`run_direct_eval.sh` 用 `PY` 环境变量指定解释器。
- **外部资产**（不在仓库内，需链接到 `baselines/x-navdp/` 下）：
  - `data/scenes` — 场景库（clutter + intern home/commercial）
  - `data/robots/{unitreeg1,dingo}.usd` — 机器人模型
  - `src/environment/controllers/checkpoints/{humanoid_g1,quadruped_go2}/policy.pt` — 步态策略
- **权重**：`checkpoints/`（见 [checkpoints/README.md](checkpoints/README.md)）— posttrain、pretrain、yolo、rl_direct_tracker。

## 使用

统一入口 `python run.py <command>`，每条工作线只有一个 `cli.py`。FM / DDIM 从仓库根运行；**RL 命令以 `baselines/x-navdp/` 为工作目录**。

### FM 蒸馏

```bash
# 1. 数采 + 标注（同时，共享编码器 joint 标注）
python run.py fm-joint-train --output <out> --scene-limit 1 --episodes 1 --physical-gpu 1 --execute

# 2. 训练（需 train+val snapshot+labels）
python run.py fm train --snapshot <snap.json> --labels <labels> --output <out> --steps N --physical-gpu 1

# 3. eval（自包含 deploy.pt，单权重闭环推理）
python run.py fm-eval --deploy <out>/deploy.pt --output <eval> --episodes N --physical-gpu 1 --execute
```

> 采集即标注，无单独离线 `label` 步。训练用 `--snapshot`/`--labels`（`merge-labels` 产物），或用 `--dataset <混合目录>`（`merge-success` 失败清洗后）直接训练。源码指纹随整理变化，旧快照/checkpoint 不能直接续跑。

### RL 执行

```bash
# 以 baselines/x-navdp/ 为工作目录
cd baselines/x-navdp

# 1. 数采（MPC 教师 BC）
python ../../rl/cli.py collect --scene-index 0 --episodes-per-scene 1 --output outputs/bc.pt

# 2. BC 初始化
python ../../rl/cli.py train-bc --data outputs/bc.pt --epochs N --output outputs/bc_policy.pt

# 3. PPO 训练
python ../../rl/cli.py train-tracker --scene-index 0 --iterations N --bc-init outputs/bc_policy.pt

# 4. eval（direct 模式）
DIRECT_CHECKPOINT=<训练输出 latest.pt> EVAL_MODE=direct bash ../../rl/scripts/eval_pointgoal.sh
```

### DDIM

```bash
python run.py ddim-benchmark --help
python run.py ddim-compare --help
python run.py ddim-sweep --help
```

## 延时基准

batch=1、8 候选、RTC on，GPU 1（RTX 4090），单位 ms：

| 方法 | 平均 (ms) |
| --- | --- |
| DDPM-10steps-baseline | 363.17 |
| DDIM-5steps | 190.85 |
| Flow Matching-4steps | 64.74 |

DDIM-5steps 是实测较稳定的 baseline 加速版本（~ 2×）；FM4 ~ 5.6×。

## 目录

```text
Re-X-NavDP/
├── baselines/x-navdp/           集成基线（环境 / 训练器 / 评估接线）
├── FM_distillation/   FM 蒸馏：cli.py + src/
├── rl/                RL 执行：cli.py + src/
├── ddim/              少步采样：cli.py + src/
├── bridge/            共享桥接（teacher / trajectory / Q / recovery）
├── rexnavdp/          路径与 bootstrap 单一来源
├── checkpoints/       外部权重索引（见 checkpoints/README.md）
├── third_party/       依赖源码与各自许可证
├── tests/             回归测试
└── run.py             入口
```

## 测试

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:baselines/x-navdp python -m unittest discover -s tests
```

## 引用与致谢

本仓库是 [NavDP](https://github.com/InternRobotics/NavDP) 及其后续工作 [X-NavDP](https://github.com/InternRobotics/NavDP/tree/master/baselines/x-navdp) 的**个人学习**：三条工作线（RL 实时执行器、FM 候选生成器蒸馏、DDIM 少步采样）都建立在 NavDP / X-NavDP 的代码、场景与预训练权重之上。

**NavDP**（基础导航扩散策略，本仓库权重与骨干的来源）

```bibtex
@inproceedings{cai2025navdp,
  title={NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
  author={Cai, Wenzhe and Peng, Jiaqi and Yang, Yuqiang and Zhang, Yujian and Wei, Meng and Wang, Hanqing and Chen, Yilun and Wang, Tai and Pang, Jiangmiao},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  eprint={2505.08712}
}
```

**X-NavDP**（本仓库蒸馏/执行基线的直接上游）

```bibtex
@misc{yang2026xnavdp,
  title={X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching},
  author={Yang, Tianyu and Zeng, Yiming and Cai, Wenzhe and Yang, Yuqiang and Peng, Jiaqi and Cheng, Hui and Pang, Jiangmiao and Wang, Tai},
  year={2026},
  eprint={2607.28560}
}
```

**致谢**：感谢 NavDP 与 X-NavDP 作者开源代码、场景与预训练权重，以及上海人工智能实验室 / InternRobotics 的相关工作。上游 X-NavDP 遵循 [MIT 许可](baselines/x-navdp/LICENSE)，[CITATION](baselines/x-navdp/CITATION.cff) 记录在案；第三方依赖见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
