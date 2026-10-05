# Checkpoints

集中存放运行时需要的外部权重（不随仓库提交）。脚本默认从本目录索引，路径统一为
`rexnavdp.CHECKPOINT_DIR / "<name>"`（即仓库根 `checkpoints/<name>`）。

需要的权重（从原 NavDP 工作区拷贝或符号链接过来）：

| 文件 | 用途 |
| --- | --- |
| `x-navdp_posttrain.ckpt` | post-trained 基线权重（FM 教师骨干 / RL 编码器 / MPC baseline） |
| `navdp_pretrained.ckpt` | 上游预训练权重（HF 文件名 `navdp_pretrain.ckpt`，需改名） |
| `yolo26n-depth.pt` | RL YOLO 深度视觉后端权重 |
| `rl_direct_tracker.pt` | RL direct 执行器权重（YOLO 深度、clutter easy+hard 混合训练产物） |

示例（链接，避免重复占用磁盘）：

```bash
ln -s /mnt/data3/cqz/nav/X-NavDP/checkpoints/x-navdp_posttrain.ckpt x-navdp_posttrain.ckpt
ln -s /mnt/data3/cqz/nav/X-NavDP/navdp_pretrain.ckpt navdp_pretrained.ckpt
ln -s <rl-direct-run>/stages/.../latest.pt rl_direct_tracker.pt
```

本目录内的 `.ckpt`/`.pt`/`.pth` 由 `.gitignore` 排除；仅此 README 被跟踪。
