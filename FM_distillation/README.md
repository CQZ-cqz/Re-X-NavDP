# Flow Matching 蒸馏

实际实现位于 `src/`（库）与 `cli.py`（统一入口）；不再保留 `x-navdp` 下的 FM 兼容入口，也没有 `scripts/` 平铺目录。导入使用 `FM_distillation.src.*`，命令使用根目录 `run.py`（或 `python -m FM_distillation.cli <stage>`）。

`src/` 按功能分模块：

- `flow_generator.py`：条件 FM、Euler 采样，候选重排调用 `bridge` 的轨迹转换和双 Q 评价。
- `fm_data.py`：观测/标签契约与 teacher tap；`fm_capture_agent.py`：闭环捕获。
- `storage.py`：不可变快照、标签加载、哈希/原子写（原 `fm_training_data`）。
- `labeling.py`：共享编码器 joint 标注；`rtc.py`：轨迹前缀 RTC 采样。
- `capture.py`（full/dual）、`joint.py`（joint/joint-train）：采集工作流。
- `training.py`：freeze/label/train/rank 基础流水线 + all-scenes/all-candidates 变体 + 验证标注。
- `evaluation.py`：闭环评估与 RTC 评估；`merging.py`：标签合并；`scheduling.py`：排队 RTC 评估。
- `dataset.py`：teacher 的 prepare/serve/evaluate/label/validate 原语。

`cli.py` 的 stage：`dataset`、`collect-full`、`collect-dual`、`collect-joint`、`collect-joint-train`、`label-validation`、`merge-labels`、`merge-success`、`train`（freeze/label/train/rank）、`train-all-scenes`、`train-all-candidates`、`label-dual`、`train-dual`、`eval-closed-loop`、`eval-rtc`、`queue`、`bench`。根目录 `run.py` 以 `fm*` 短名转发到这些 stage（如 `python run.py fm --help`、`python run.py fm-eval --help`、`python run.py fm-collect --help`）。

基础离线蒸馏学习教师无 RTC 的 8 候选分布，原增量作为目标，不是平滑后的累计路径。后来新增的 joint/RTC 路径保留为显式实验变体，不悄悄替代该协议。具体开关以每个 CLI 帮助与 run 元数据为准。训练/标注产物不在源码目录提交；旧工作区数据指纹不会在新仓库里自动兼容。

## 网络与损失

- **学生**（`CompactFlowGenerator`，`flow_generator.py`）：条件 Flow Matching 生成器，只替换教师的扩散解码器。共享并冻结教师的条件模块（`cond_pos_embed`/`out_pos_embed`/`time_emb`/`embodiment_embedding` 等），新初始化 4 层 `TransformerDecoder` + 速度头 `Linear(384→3)`。前向为 `x(t)=(1-t)·noise + t·target`，预测速度 `v=target-noise`。
- **教师骨干**（`FrozenBackbone`，`fm_backbone.py`）：冻结的 RGB-D 编码器（DepthAnythingV2）+ goal 编码 + 双 Q 评价 + B-spline 平滑，训练时提供条件特征与目标，推理时做候选重排。
- **损失**：flow matching MSE `‖v_θ(x_t,t,c)−(target−noise)‖²`。`noise` 默认取标签里存的教师 `initial_noise`（**latent 对齐**：`z_i→τ_i` 一一配对，而非随机重采样），保住 mode identity；可选叠加对称 distribution loss（见下）。`q1/q2/scores` 已存标签，但只在推理重排时用冻结双 Q 评分，不参与训练 loss、不反传 Q。

### 分布塌缩、latent 对齐与 distribution loss

离线蒸馏存在**多样性塌缩（单峰化）**：MSE 回归把 8 个多峰目标往均值抹平，学生 `student_pairwise_ade_m` 长期只有教师（~0.34）的一半，且 `fm_loss` 继续下降、塌缩还在加剧——loss 不感知这个退化。两个正交的修复：

1. **latent 对齐**（`training.py`，默认开启）：`all_candidate_loss`/`flow_loss` 用标签里的 `initial_noise` 作为 `z_i`，与对应 `τ_i` 配对学习直线映射 `z_i→τ_i`，而不是 `torch.randn` 随机配对。代价为零、不改变 FM 目标；只解决「谁对应谁」，不单独解决塌缩。

2. **对称 distribution loss**（`training.distribution_loss`，`train-all-candidates --dist-lambda-mu/--dist-lambda-sigma` 开启）：在累计路径上匹配 student 与 teacher 的**均值**（位置）和**每航点标准差**（离散度）：

```
L_dist = λ_μ·‖μ_S−μ_T‖² + λ_σ·‖std_S−std_T‖²    # 都对称，双向罚
```

双向罚使学生在太窄（塌缩）和太宽（overshoot）时都被拉回，配合 latent 对齐把多样性从 ~0.15 稳定提到 ~0.27（教师 0.34），且不失控。采样经可微的 `sample_with_grad`（Euler ODE）反传。单边 hinge（`max(0, teacher−student)`）会 overshoot 到 0.53 且散错方向，已弃用。

### 双分支蒸馏 / condition dropout（`fm_dual.py`，实验记录，未生效）

`train-dual` 在 v1 标签上做 condition dropout（4 真实 goal + 4 零 token，回归同一批 pointgoal 轨迹），动机是保留 goal 无关先验、防塌缩。**实测与 8 候选等权打平（~0.16），未缓解塌缩**，故保留为实验记录、不作主路径：

```
L = β·L_pg + α·L_ng
L_pg = Σ_k w_k · ℓ_k，  w_k = (1−λ)/K + λ·softmax(q̃_k/T)   # 温和 Q 加权，不反传 Q
L_ng = (1/K)·Σ_k ℓ_k                                    # 等权
```

`label-dual` 是另一条已弃用变体（4 pointgoal + 4 零 token **目标** 标注，`validate_dual_label`）。部署/闭环推理走 pointgoal 分支。

## 自包含部署权重

蒸馏只替换了扩散解码器，全链路推理仍需教师冻结的 RGB-D 编码器、goal 编码器、双 Q 评价和 B-spline 平滑。为避免推理时同时加载 posttrain + student 两份权重，`src/fm_backbone.py` 把所需的教师骨干（`FrozenBackbone`）拷进了 FM，训练结束时会导出**一份** `deploy.pt`（`student` + 教师骨干 `state_dict` + `backbone_meta` + `signature`）。

`FMPolicy`（同文件）加载这份 `deploy.pt` 即可完成 encode→sample→rank 的全链路离线推理：

```python
from FM_distillation.src.fm_backbone import FMPolicy
policy = FMPolicy("outputs/<run>/deploy.pt", device="cuda:0")
result = policy.predict(goal, rgb, depth, embodiment=1)  # trajectories / scores / top_trajectories
```

部署权重不再依赖 `baselines/x-navdp/eval/src` 的 posttrain 权重，也不依赖两份权重同时驻留。

闭环评估可直接用这份单权重替代「student + posttrain」两份加载，通过 `--deploy` 传入：

```bash
python run.py fm-eval --deploy outputs/<run>/deploy.pt --output /path/to/eval --execute
python run.py fm-rtc-eval --deploy outputs/<run>/deploy.pt --output /path/to/eval --rtc on --execute
```

`--deploy` 与 `--student` 互斥；`--checkpoint`（posttrain）此时仅用于运行目录的 teacher 指纹校验（`dataset.prepare`/`check_frozen`），推理时只加载 `deploy.pt`。
