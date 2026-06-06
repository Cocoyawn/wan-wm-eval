# wan-wm-eval

在 **tower-of-hanoi** val-data 上评估 Wan2.2-TI2V-5B 世界模型的脚本集合：自回归 rollout 生成视频 + PSNR 指标 + 结果分析。

> 依赖训练/推理框架 [`Shirk6/diffsynth-studio-rlinf`](https://github.com/Shirk6/diffsynth-studio-rlinf)：脚本 `from diffsynth ...` 直接复用其 pipeline / VAE / dataset。运行前需把该仓库根目录加入 `PYTHONPATH`。

---

## 1. 这些脚本分别是什么

### rollout/ —— 整条轨迹自回归生成视频

世界模型一次只能生成一个 57 帧窗口（9 帧 condition + 48 帧 predict）。要生成整条轨迹（几百到上千帧），必须**一段接一段**：把上一段的结尾接成下一段的 condition。三个脚本的**唯一区别就是「怎么接」**，这也是这套 eval 的核心研究点。

| 脚本 | 接力方式 | 状态 / 效果 |
|---|---|---|
| `rollout_pixel_feedback.py` | **像素接力**：上一段 decode 成 9 帧像素 → 作为下一段 condition，pipe 内部重新 encode | ✅ 可用，整图 PSNR ≈ **12.5dB**。缺点：condition 第一帧始终钉死轨迹首帧；每段 decode→encode 引入 VAE 重建噪声 |
| `rollout_latent_naive.py` | **latent 直传**：直接切上一段去噪后 latent 的尾部 3 帧，塞进下一段 condition，完全不过 pixel | ⚠️ **有已知 bug**：偏色漂移，PSNR ≈ **10dB**（反而更差）。根因见下方「踩坑记录」。保留作对照 |
| `rollout_latent_reanchor.py` | **latent + 因果重锚定**：condition 的「开机帧」(latent[0]) 用上一段 pixel[48] 重新 encode 拿到正确分布，其余 2 帧 latent 直传 | 🚧 最新方案，理论上最对（VAE 噪声只从 1 帧进）。**仍在调试**，未定稿 |

### metrics/ —— 单窗口指标（不生成整条视频）

| 脚本 | 算什么 |
|---|---|
| `window_denoise_psnr.py` | 按训练同样的滑窗，对每个窗口算：① **去噪 loss**（复用 `WanTrainingModule.training_loss`，与训练 val_loss 同口径）② **teacher-forcing PSNR**（喂真 condition 单步采样，不累积误差，反映单步预测上限，≈ 21dB） |

### analysis/ —— 结果分析（不跑模型，只处理产出的 json/mp4）

| 脚本 | 做什么 |
|---|---|
| `summarize_psnr.py` | 合并多卡分片结果 → 总 PSNR 均值 + `summary.csv` |
| `plot_decay_and_groups.py` | 画 PSNR 随帧衰减曲线 + 按 failure/success 分组统计柱状图 |
| `compare_two_ckpts.py` | 两个 ckpt 配对对比：衰减叠加 / 分组柱状 / 逐条差异直方图 |
| `make_sidebyside_video.py` | 拼 `GT ｜ 生成` 并排对比视频 |

### legacy/ —— 废弃，仅复现旧结果

| 脚本 | 说明 |
|---|---|
| `rollout_buggy_hack.py` | 最早一版，含 `actions[0,-1]=-1` 等训练里不存在的 action hack。已废弃，只为复现历史评估数据 |

---

## 2. 快速开始

```bash
# 1) 准备依赖框架
git clone https://github.com/Shirk6/diffsynth-studio-rlinf.git
export PYTHONPATH=/path/to/diffsynth-studio-rlinf:$PYTHONPATH

# 2) 改脚本顶部的 DATA_ROOT / VAE_PATH / ckpt 路径为你的本地路径
#    (或用命令行 --ckpt 覆盖)

# 3) 单条跑通(随机抽一条 val)
python rollout/rollout_pixel_feedback.py \
    --device cuda:0 --ckpt /path/to/step-42000.safetensors \
    --limit 1 --out_root outputs/probe --save_gt

# 4) 4 卡全量(86 条)
bash run_4gpu.sh /path/to/step-42000.safetensors 5 outputs/rollout_full

# 5) 汇总 + 出图
python analysis/summarize_psnr.py --out_root outputs/rollout_full
python analysis/plot_decay_and_groups.py --out_root outputs/rollout_full
```

`rollout_pixel_feedback.py` 和 `rollout_latent_naive.py` 支持 `--shard i --num_shards N` 多卡分片（按 `idx % N == i` 切分 86 条），可直接喂给 `run_4gpu.sh`。

`rollout_latent_reanchor.py` 目前是**单条 probe 脚本**（仍在调试），用法不同：

```bash
# 随机抽一条 val 跑因果重锚定版, 输出 GT/生成 mp4 + RGB 漂移检验
python rollout/rollout_latent_reanchor.py \
    --device cuda:0 --ckpt /path/to/step-42000.safetensors \
    --seed_pick 42 --out_dir outputs/reanchor_probe
```

---

## 3. 关键配置（对齐训练脚本 `Wan2.2-TI2V-5B_rlinf.sh`）

| 参数 | 值 | 说明 |
|---|---|---|
| 分辨率 | 544 × 320 | 三视角竖直拼接（cam_high / cam_left_wrist / cam_right_wrist，各 180 行） |
| num_frames | 57 | 一个窗口 = 9 condition + 48 predict |
| condition_frames | 9 | → VAE 因果压缩成 3 个 latent 帧 |
| action_dim | 14 | 双臂，每臂 7 维 |
| Ta / To | 48 / 8 | action 窗口；`retain_actions=True` |
| 去噪步数 | 5 | 推理 `num_inference_steps`；cfg_scale=1.0 |

PSNR 口径：整图 + 分三视角逐帧，对齐到 `min(生成, GT)` 帧。

---

## 4. 踩坑记录：为什么 latent 直传会偏色（latent-AR 核心坑）

Wan VAE 的**时间维是因果的**，57 帧 → 15 个 latent 帧不是均匀的：

- **latent[0]（「开机帧」）**：VAE 用 `'Rep'`（自身复制）当因果 padding，**单帧**编码 → 分布特殊（实测 std ≈ 0.73）
- **latent[k≥1]（「播放帧」）**：每个对应 4 帧像素，依赖前一帧的 `feat_cache` → 分布不同（std ≈ 1.13）

`rollout_latent_naive.py` 的 bug：把上一段的「播放帧」latent（std≈1.13）直接塞进下一段 condition 的**开机帧槽位**（模型期望 std≈0.73）。分布错配逐段累积 → 偏色/发紫。

`rollout_pixel_feedback.py` 之所以没这问题：它把上一段 decode 成像素再 encode，VAE 重新走一遍因果流程，**自动把第一帧重新锚定成正确的「开机帧」latent**——这次 decode→encode 顺手做了「重新因果锚定」这件正事（代价是引入重建噪声）。

`rollout_latent_reanchor.py` 的修法：只把开机帧那 1 帧重新 encode（拿到正确分布），另 2 帧 latent 直传 → 噪声只从 1 帧进，又避免槽位错配。

---

## 5. val 清单

`val_list.txt`：86 条验证轨迹相对路径（30 failure-data + 56 success-and-hil-data）。
