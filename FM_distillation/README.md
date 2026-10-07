# Flow Matching 蒸馏

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


## 网络与损失

- **学生**（`CompactFlowGenerator`，`flow_generator.py`）：条件 Flow Matching Generator，只替换教师的扩散解码器；冻结教师条件模块，新初始化 4 层 `TransformerDecoder` + 速度头。前向 `x(t)=(1-t)·noise + t·target`，预测速度 `v=target-noise`。
- **教师骨干**（`FrozenBackbone`，`fm_backbone.py`）：冻结的 RGB-D 编码器（DepthAnythingV2）+ goal 编码 + 双 Q 评价 + B-spline 平滑。
- **损失**：flow matching MSE `‖v_θ(x_t,t,c)−(target−noise)‖²`，`noise` 默认用标签里存的教师 `initial_noise`（**latent 对齐**：`z_i→τ_i` 配对）。为对抗多样性塌缩，可选叠加：
  - **对称 distribution loss**（`--dist-lambda-mu/--dist-lambda-sigma`）：`L = λ_μ‖μ_S−μ_T‖² + λ_σ‖std_S−std_T‖²`
  - **Sinkhorn OT loss**（`--sinkhorn-lambda/--sinkhorn-eps/--sinkhorn-iters`）：`L = <P,C>`，熵正则 OT 匹配 on-policy 采样与教师候选的完整分布

  `q1/q2/scores` 已存标签，但只在推理重排时用冻结双 Q 评分，不参与训练、不反传 Q。

### 双分支蒸馏（已弃用）

沿用 x-navdp 的 condition dropout：4 pointgoal + 4 nogoal，`L = β·L_pg + α·L_ng`（pointgoal 温和 Q 加权、nogoal 等权）。latent 对齐 + distribution loss 加入后未再启用。


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
