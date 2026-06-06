"""
逐窗口指标 eval —— 对齐训练逻辑 (train_rlinf.py)

对 val-data 的每条轨迹按训练同样的滑窗，计算两个指标：
  1. 去噪 loss：复用 WanTrainingModule.forward(=training_loss)，与训练 val_loss 同口径。
     timestep 随机，故每个窗口多次采样取均值降噪。
  2. teacher-forcing PSNR：对该窗口一次性采样(condition=窗口首帧+context, action=窗口action)，
     decode 后与该窗口 GT 帧比 PSNR（整图+分视角）。不累积误差，反映单步预测能力。
"""
import os
os.environ["WAN_ACTION_DIM"] = "14"
os.environ["WAN_CONDITION_FRAMES"] = "9"

import sys
import json
import time
import argparse

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_training"))
from train_rlinf import WanTrainingModule
from diffsynth.trainers.dataset import RLinfDataset

VAE_PATH = "/path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
VAL_BASE = "/path/to/Challenge-phase1-dataset-rlinf/tower-of-hanoi-game/val-data"
VIEW_BOUNDS = {"cam_high": (0, 180), "cam_left_wrist": (180, 360), "cam_right_wrist": (360, 540)}

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--val_base", type=str, default=VAL_BASE)
parser.add_argument("--out_root", type=str, default="outputs/eval_window_metrics")
parser.add_argument("--loss_repeat", type=int, default=8, help="每窗口随机 timestep 采样次数(降噪)")
parser.add_argument("--tf_psnr", action="store_true", help="是否额外算 teacher-forcing PSNR(慢)")
parser.add_argument("--tf_steps", type=int, default=5)
parser.add_argument("--max_windows_per_traj", type=int, default=4, help="每条轨迹采样多少个窗口(0=全部)")
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(f"[{args.device}] 加载 ckpt={args.ckpt}")
model = WanTrainingModule(
    model_paths=json.dumps([args.ckpt, VAE_PATH]),
    trainable_models="",                 # eval 不训练
    extra_inputs="input_image,action",
    static_video_prob=0.0,
    action_dim=14,
)
model.pipe.device = args.device
model.pipe.dit.to(args.device)
model.pipe.vae.to(args.device)
model.eval()


def _to_u8_hwc(img):
    a = np.asarray(img.convert("RGB"), dtype=np.uint8) if isinstance(img, Image.Image) else np.asarray(img)
    return a


def _psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse <= 1e-12 else float(20*np.log10(255.0) - 10*np.log10(mse))


def view_psnr(gen, gt):
    out = {"full": _psnr(gen, gt)}
    for name, (lo, hi) in VIEW_BOUNDS.items():
        out[name] = _psnr(gen[lo:hi], gt[lo:hi])
    return out


@torch.no_grad()
def window_loss(data, repeat):
    """多次随机 timestep 取均值，复用训练 forward(=training_loss)。"""
    losses = []
    inputs = model.forward_preprocess(data)
    for _ in range(repeat):
        loss = model.pipe.training_loss(
            **{n: getattr(model.pipe, n) for n in model.pipe.in_iteration_models},
            **inputs,
        )
        losses.append(float(loss.item()))
    return float(np.mean(losses)), float(np.std(losses))


def main():
    ds = RLinfDataset(base_path=[args.val_base], Ta=48, To=8,
                      retain_actions=True, action2obs_bias=False, action_dim=14)
    # sample_indices: [(episode_idx, env_id, start), ...]
    by_ep = {}
    for gi, (ep, env, start) in enumerate(ds.sample_indices):
        by_ep.setdefault(ep, []).append(gi)
    eps = sorted(by_ep.keys())
    if args.limit:
        eps = eps[:args.limit]
    eps = [e for i, e in enumerate(eps) if i % args.num_shards == args.shard]

    os.makedirs(args.out_root, exist_ok=True)
    results = []
    t0 = time.time()
    for ep in eps:
        gis = by_ep[ep]
        if args.max_windows_per_traj > 0:
            sel = np.linspace(0, len(gis)-1, min(args.max_windows_per_traj, len(gis))).astype(int)
            gis = [gis[i] for i in sel]
        ep_path = ds.episode_info[ep][0]
        name = os.path.basename(ep_path.rstrip("/"))
        win_losses, tf_list = [], []
        for gi in gis:
            data = ds[gi]
            lo_mean, lo_std = window_loss(data, args.loss_repeat)
            win_losses.append(lo_mean)
            if args.tf_psnr:
                tf_list.append(teacher_forcing_psnr(data))
        rec = {"name": name, "episode": int(ep), "n_windows": len(gis),
               "denoise_loss": float(np.mean(win_losses))}
        if tf_list:
            rec["tf_psnr_full"] = float(np.mean([t["full"] for t in tf_list]))
            for v in VIEW_BOUNDS:
                rec[f"tf_psnr_{v}"] = float(np.mean([t[v] for t in tf_list]))
        results.append(rec)
        msg = f"[{name}] loss={rec['denoise_loss']:.4f}"
        if tf_list:
            msg += f" tf_psnr={rec['tf_psnr_full']:.2f}dB"
        print(msg, flush=True)

    out = os.path.join(args.out_root, f"_shard_{args.shard}_of_{args.num_shards}.json")
    json.dump({"device": args.device, "ckpt": args.ckpt, "shard": args.shard,
               "num_shards": args.num_shards, "seconds": time.time()-t0,
               "results": results}, open(out, "w"), indent=2)
    print(f"[{args.device}] 完成 {len(results)} 条 -> {out}  用时 {time.time()-t0:.1f}s")


@torch.no_grad()
def teacher_forcing_psnr(data):
    """单窗口一次性采样，与 GT 窗口比 PSNR。"""
    gt_frames = data["video"]   # List[PIL] 57 帧
    h, w = gt_frames[0].size[1], gt_frames[0].size[0]
    out_video = model.pipe(
        input_image=gt_frames[0],
        input_image4=gt_frames[1:9],          # 训练 condition=9 帧(首帧+8 context)
        action=data["action"].to(args.device),
        height=h, width=w, num_frames=len(gt_frames),
        num_inference_steps=args.tf_steps, cfg_scale=1.0, tiled=False,
    )
    gen = out_video[0]
    n = min(len(gen), len(gt_frames))
    accs = {"full": [], **{v: [] for v in VIEW_BOUNDS}}
    for i in range(n):
        g = _to_u8_hwc(gen[i]); t = _to_u8_hwc(gt_frames[i])
        if g.shape[:2] != t.shape[:2]:
            g = np.asarray(Image.fromarray(g).resize((t.shape[1], t.shape[0])), dtype=np.uint8)
        vp = view_psnr(g, t)
        for k in accs: accs[k].append(vp[k])
    return {k: float(np.mean([x for x in vs if np.isfinite(x)])) for k, vs in accs.items()}


if __name__ == "__main__":
    main()
