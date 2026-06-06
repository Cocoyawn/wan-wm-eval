# wan-wm-eval

在 tower-of-hanoi val-data 上评估 Wan2.2-TI2V-5B 世界模型：自回归 rollout 生成视频 + PSNR 指标 + 结果分析。

依赖 [`Shirk6/diffsynth-studio-rlinf`](https://github.com/Shirk6/diffsynth-studio-rlinf)（脚本复用其 pipeline / VAE / dataset），运行前把该仓库根目录加入 `PYTHONPATH`。

## 快速开始

```bash
# 1) 依赖框架
git clone https://github.com/Shirk6/diffsynth-studio-rlinf.git
export PYTHONPATH=/path/to/diffsynth-studio-rlinf:$PYTHONPATH

# 2) 改脚本顶部 DATA_ROOT / VAE_PATH / ckpt 路径(或用 --ckpt 覆盖)

# 3) 单条跑通
python rollout/rollout_pixel_feedback.py \
    --device cuda:0 --ckpt /path/to/step-42000.safetensors \
    --limit 1 --out_root outputs/probe --save_gt

# 4) 4 卡全量(86 条)
bash run_4gpu.sh rollout/rollout_pixel_feedback.py /path/to/step-42000.safetensors 5 outputs/rollout_full

# 5) 汇总 + 出图
python analysis/summarize_psnr.py --out_root outputs/rollout_full
python analysis/plot_decay_and_groups.py --out_root outputs/rollout_full
```

`rollout/` 下三个脚本支持 `--shard i --num_shards N` 多卡分片（按 `idx % N == i` 切分）。

## 关键配置（对齐训练脚本 `Wan2.2-TI2V-5B_rlinf.sh`）

| 参数 | 值 | 说明 |
|---|---|---|
| 分辨率 | 544 × 320 | 三视角竖直拼接（cam_high / cam_left_wrist / cam_right_wrist，各 180 行） |
| num_frames | 57 | 一个窗口 = 9 condition + 48 predict |
| condition_frames | 9 | VAE 因果压缩成 3 个 latent 帧 |
| action_dim | 14 | 双臂，每臂 7 维 |
| Ta / To | 48 / 8 | action 窗口；`retain_actions=True` |
| 去噪步数 | 5 | `num_inference_steps`；cfg_scale=1.0 |

PSNR：整图 + 分三视角逐帧，对齐到 `min(生成, GT)` 帧。
`val_list.txt`：86 条验证轨迹（30 failure + 56 success-and-hil）。
