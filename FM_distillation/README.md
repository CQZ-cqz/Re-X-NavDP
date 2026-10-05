# Flow Matching 蒸馏

实际实现位于 `core/`（库）与 `cli.py`（统一入口）；不再保留 `x-navdp` 下的 FM 兼容入口，也没有 `scripts/` 平铺目录。导入使用 `FM_distillation.core.*`，命令使用根目录 `run.py`（或 `python -m FM_distillation.cli <stage>`）。

`core/` 按功能分模块：

- `flow_generator.py`：条件 FM、Euler 采样，候选重排调用 `bridge` 的轨迹转换和双 Q 评价。
- `fm_data.py`：观测/标签契约与 teacher tap；`fm_capture_agent.py`：闭环捕获。
- `storage.py`：不可变快照、标签加载、哈希/原子写（原 `fm_training_data`）。
- `labeling.py`：共享编码器 joint 标注；`rtc.py`：轨迹前缀 RTC 采样。
- `capture.py`（full/dual）、`joint.py`（joint/joint-train）：采集工作流。
- `training.py`：freeze/label/train/rank 基础流水线 + all-scenes/all-candidates 变体 + 验证标注。
- `evaluation.py`：闭环评估与 RTC 评估；`merging.py`：标签合并；`scheduling.py`：排队 RTC 评估。
- `dataset.py`：teacher 的 prepare/serve/evaluate/label/validate 原语。

`cli.py` 的 stage：`dataset`、`collect-full`、`collect-dual`、`collect-joint`、`collect-joint-train`、`label-validation`、`merge-labels`、`merge-success`、`train`（freeze/label/train/rank）、`train-all-scenes`、`train-all-candidates`、`eval-closed-loop`、`eval-rtc`、`queue`、`bench`。根目录 `run.py` 以 `fm*` 短名转发到这些 stage（如 `python run.py fm --help`、`python run.py fm-eval --help`、`python run.py fm-collect --help`）。

基础离线蒸馏学习教师无 RTC 的 8 候选分布，原增量作为目标，不是平滑后的累计路径。后来新增的 joint/RTC 路径保留为显式实验变体，不悄悄替代该协议。具体开关以每个 CLI 帮助与 run 元数据为准。训练/标注产物不在源码目录提交；旧工作区数据指纹不会在新仓库里自动兼容。

## 自包含部署权重

蒸馏只替换了扩散解码器，全链路推理仍需教师冻结的 RGB-D 编码器、goal 编码器、双 Q 评价和 B-spline 平滑。为避免推理时同时加载 posttrain + student 两份权重，`core/fm_backbone.py` 把所需的教师骨干（`FrozenBackbone`）拷进了 FM，训练结束时会导出**一份** `deploy.pt`（`student` + 教师骨干 `state_dict` + `backbone_meta` + `signature`）。

`FMPolicy`（同文件）加载这份 `deploy.pt` 即可完成 encode→sample→rank 的全链路离线推理：

```python
from FM_distillation.core.fm_backbone import FMPolicy
policy = FMPolicy("outputs/<run>/deploy.pt", device="cuda:0")
result = policy.predict(goal, rgb, depth, embodiment=1)  # trajectories / scores / top_trajectories
```

部署权重不再依赖 `x-navdp/eval/src` 的 posttrain 权重，也不依赖两份权重同时驻留。

闭环评估可直接用这份单权重替代「student + posttrain」两份加载，通过 `--deploy` 传入：

```bash
python run.py fm-eval --deploy outputs/<run>/deploy.pt --output /path/to/eval --execute
python run.py fm-rtc-eval --deploy outputs/<run>/deploy.pt --output /path/to/eval --rtc on --execute
```

`--deploy` 与 `--student` 互斥；`--checkpoint`（posttrain）此时仅用于运行目录的 teacher 指纹校验（`dataset.prepare`/`check_frozen`），推理时只加载 `deploy.pt`。
