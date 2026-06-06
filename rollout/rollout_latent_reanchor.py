"""
方案 (C-v2) 修正版纯 latent 自回归 —— 因果重锚定
=========================================================
根因(已用 diag_latent_drift.py 证实): Wan VAE 时间维因果, latent[0] 是
"开机帧"(Rep 因果 padding, 单帧编码, std≈0.73), latent[k>=1] 是"播放帧"
(每4帧一组, 依赖前帧 feat_cache, std≈1.13)。v1 把上一段的"播放帧"latent
直接塞进新窗口的开机帧槽 0 -> 分布错配 -> 偏色漂移。

修法: 新窗口 condition = latent[0,1,2]:
  - latent[0] (开机帧) = 上一段 pixel[48] 重新走 VAE encode -> 正确分布的开机帧
  - latent[1] = 上一段 slot13 (=pixel[49..52]) 纯 latent 直传
  - latent[2] = 上一段 slot14 (=pixel[53..56]) 纯 latent 直传
=> VAE round-trip 噪声只从 1 帧进, 且开机帧槽位分布正确。

实现: monkey-patch ImageEmbedderFused.process, 当 pipe._cond_latent3 已注入,
直接用它(已是组装好的 3 帧 condition latent), 跳过内部 encode。
我们在外部自己 encode 开机帧 + 拼接直传 latent。
"""
import os
os.environ["WAN_ACTION_DIM"] = "14"
os.environ["WAN_CONDITION_FRAMES"] = "9"

import json, time, argparse
import numpy as np, torch
from PIL import Image

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth import save_video

GEN_H, GEN_W = 544, 320
CONDITION_FRAMES, PREDICT_FRAMES = 9, 48
COND_LATENT = (CONDITION_FRAMES - 1) // 4 + 1          # 3
WINDOW = CONDITION_FRAMES + PREDICT_FRAMES             # 57
VAE_PATH = "/mnt/afs-h200/yuyangcheng/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
DATA_ROOT = "/mnt/afs-h200/yuyangcheng/data/Challenge-phase1-dataset-rlinf"
VIEW_BOUNDS = {"cam_high": (0, 180), "cam_left_wrist": (180, 360), "cam_right_wrist": (360, 540)}

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt", type=str,
    default="/mnt/afs-h200/yuyangcheng/data/wan64-rlinf-cache8k-resume1k/checkpoints/step-42000.safetensors")
parser.add_argument("--rel", type=str, default="", help="单条相对路径; 空则从 list 随机取")
parser.add_argument("--list_file", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "val_list.txt"))
parser.add_argument("--seed_pick", type=int, default=42, help="随机抽条用的种子")
parser.add_argument("--out_dir", type=str, default="outputs/latent_v2_probe")
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()

print(f"[{args.device}] 加载 ckpt={args.ckpt}")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device=args.device,
    model_configs=[ModelConfig(path=args.ckpt, offload_device="cpu"),
                   ModelConfig(path=VAE_PATH, offload_device="cpu")])
pipe.dit.to(args.device); pipe.vae.to(args.device)

# ============ monkey-patch: 注入组装好的 3 帧 condition latent ============
pipe._cond_latent3 = None     # [1,C,3,Hl,Wl] 外部组装好的 condition (开机encode + 2播放直传)
pipe._last_latent = None      # decode 前截获完整去噪 latent

for u in pipe.units:
    if u.__class__.__name__ == "WanVideoUnit_ImageEmbedderFused":
        _img_unit = u
        _orig_process = u.__class__.process
        break

def patched_process(self, pipe, input_image, latents, height, width, tiled, tile_size, tile_stride, input_image4):
    if getattr(pipe, "_cond_latent3", None) is not None:
        z = pipe._cond_latent3.to(dtype=latents.dtype, device=latents.device)
        T0 = z.shape[2]
        latents[:, :, :T0] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}
    return _orig_process(self, pipe, input_image, latents, height, width, tiled, tile_size, tile_stride, input_image4)
_img_unit.__class__.process = patched_process

_orig_decode = pipe.vae.decode
def patched_decode(hidden_states, *a, **k):
    pipe._last_latent = hidden_states.detach().clone()
    return _orig_decode(hidden_states, *a, **k)
pipe.vae.decode = patched_decode
# =========================================================================


def load_gt(folder):
    rgb = np.load(os.path.join(folder, "rgb.npy"))
    ak = np.load(os.path.join(folder, "actions.npy"))
    if rgb.ndim == 5: rgb = rgb[:, 0]
    if ak.ndim == 3: ak = ak[:, 0]
    vid = []
    for f in rgb:
        im = f if f.max() > 1 else (f*255).clip(0,255).astype(np.uint8)
        vid.append(Image.fromarray(np.transpose(im, (1,2,0)).astype(np.uint8)))
    return vid, ak


def build_action_window(actions, start, Ta=48, To=8):
    dim = actions.shape[1]
    a_s, a_e = start - To + 1, start + Ta + 1
    if a_s < 0:
        pad = np.zeros((To, dim), dtype=actions.dtype)
        win = np.concatenate([pad, actions[:Ta]], axis=0)
    else:
        idx = np.clip(np.arange(a_s, a_e), 0, len(actions)-1)
        win = actions[idx]
    win = np.concatenate([np.zeros((1, dim), dtype=actions.dtype), win], axis=0)
    return win


@torch.no_grad()
def encode_single_pixel(img_pil):
    """单帧 pixel -> 开机帧 latent [1,C,1,Hl,Wl] (走 VAE 因果 encode, Rep padding)"""
    vid = pipe.preprocess_video([[img_pil.resize((GEN_W, GEN_H))]])   # [1,3,1,H,W]
    z = pipe.vae.encode(vid, device=args.device, tiled=False)         # [1,C,1,Hl,Wl]
    return z.to(dtype=torch.bfloat16, device=args.device)


@torch.no_grad()
def latent_rollout_v2(rgb_list, actions, steps):
    actions = np.asarray(actions)
    action_len = len(actions)
    num_iters = max(1, (action_len - 1 + PREDICT_FRAMES - 1) // PREDICT_FRAMES)
    print(f"动作帧数={action_len} chunks={num_iters} (latent-v2 因果重锚定)")

    first_frame = rgb_list[0]
    all_pred_latents = []
    first_chunk_head = None        # 首 chunk 的 latent[0:3] (含真开机帧)
    prev_full = None               # 上一段完整去噪 latent [1,C,15,Hl,Wl]
    prev_anchor_pixel = None       # 上一段 pixel[48] (下段开机帧来源)

    for i in range(num_iters):
        print(f"--- Chunk {i+1}/{num_iters} ---")
        start = i * PREDICT_FRAMES
        act = build_action_window(actions, start, Ta=PREDICT_FRAMES, To=CONDITION_FRAMES-1)
        act = torch.from_numpy(np.ascontiguousarray(act)).to(dtype=torch.bfloat16, device=args.device)

        if i == 0:
            pipe._cond_latent3 = None    # 首 chunk: GT 首帧+8context 原生 encode
            _ = pipe(seed=0, tiled=False, input_image=first_frame,
                     input_image4=[first_frame]*(CONDITION_FRAMES-1),
                     action=act, height=GEN_H, width=GEN_W, num_frames=WINDOW,
                     num_inference_steps=steps, cfg_scale=1.0)
            full = pipe._last_latent
            first_chunk_head = full[:, :, :COND_LATENT]
            all_pred_latents.append(full[:, :, COND_LATENT:])
        else:
            # ---- 组装 condition latent[0,1,2] ----
            # latent[0] 开机帧 = 上一段 pixel[48] 重新 encode
            z0 = encode_single_pixel(prev_anchor_pixel)                 # [1,C,1,..]
            # latent[1,2] = 上一段 slot13,14 直传 (播放帧)
            z12 = prev_full[:, :, -2:].to(dtype=torch.bfloat16)         # [1,C,2,..]
            pipe._cond_latent3 = torch.cat([z0, z12], dim=2)           # [1,C,3,..]
            _ = pipe(seed=0, tiled=False, input_image=first_frame,      # 占位, 被 patch 跳过
                     input_image4=[first_frame]*(CONDITION_FRAMES-1),
                     action=act, height=GEN_H, width=GEN_W, num_frames=WINDOW,
                     num_inference_steps=steps, cfg_scale=1.0)
            full = pipe._last_latent
            all_pred_latents.append(full[:, :, COND_LATENT:])

        prev_full = full
        # 本段 pixel[48] = 下一段 condition 的开机帧。
        # decode 必须用【整段】latent(因果历史完整), 不能孤立 decode 单个 slot
        # (孤立 decode 会用 Rep 假历史 -> 取出的像素本身漂移)。
        # 只 re-encode 这 1 帧, VAE 噪声仍只进 1 帧。
        dec_full = _orig_decode(full.to(args.device), device=args.device, tiled=False)
        prev_pixels = pipe.vae_output_to_video(dec_full)[0]   # 57 帧
        prev_anchor_pixel = prev_pixels[48]                   # pixel[48] = 下段开机帧

    pipe._cond_latent3 = None
    full_latent = torch.cat([first_chunk_head] + all_pred_latents, dim=2)
    pipe.load_models_to_device(["vae"])
    video = _orig_decode(full_latent, device=args.device, tiled=False)
    frames = pipe.vae_output_to_video(video)[0][:action_len]
    print(f"最终生成帧数={len(frames)}")
    return frames


def _u8(img):
    return np.asarray(img.convert("RGB"), dtype=np.uint8) if isinstance(img, Image.Image) else np.asarray(img)

def _psnr(a, b):
    a=a.astype(np.float64); b=b.astype(np.float64); mse=np.mean((a-b)**2)
    return float("inf") if mse<=1e-12 else float(20*np.log10(255.)-10*np.log10(mse))

def compute_psnr(gen, gt):
    n=min(len(gen),len(gt)); full=[]; views={v:[] for v in VIEW_BOUNDS}
    for i in range(n):
        g=_u8(gen[i]); t=_u8(gt[i])
        full.append(_psnr(g,t))
        for v,(lo,hi) in VIEW_BOUNDS.items(): views[v].append(_psnr(g[lo:hi],t[lo:hi]))
    mean=lambda xs:float(np.mean([x for x in xs if np.isfinite(x)]))
    return {"num_frames":n,"full_mean":mean(full),"view_means":{v:mean(views[v]) for v in VIEW_BOUNDS},"full_pf":full}

def rgb_mean(img):
    return np.asarray(img.convert("RGB"),dtype=np.float64).reshape(-1,3).mean(0)


if __name__ == "__main__":
    if args.rel:
        rel = args.rel
    else:
        items=[l.strip() for l in open(args.list_file) if l.strip()]
        rng=np.random.RandomState(args.seed_pick)
        rel=items[rng.randint(len(items))]
    print(f"抽到: {rel}")
    folder=os.path.join(DATA_ROOT, rel)
    rgb_list, actions = load_gt(folder)

    t0=time.time()
    gen=latent_rollout_v2(rgb_list, actions, args.steps)
    sec=time.time()-t0
    psnr=compute_psnr(gen, rgb_list)

    os.makedirs(args.out_dir, exist_ok=True)
    parts=rel.strip("/").split("/"); name=f"{parts[1]}__{parts[-1]}"
    save_video(gen, os.path.join(args.out_dir, f"{name}_v2.mp4"), fps=30, quality=5)
    save_video(rgb_list[:len(gen)], os.path.join(args.out_dir, f"{name}_gt.mp4"), fps=30, quality=5)

    # 漂移检验: 生成 vs GT 首/中/末帧 RGB 均值
    idxs=[0, len(gen)//2, len(gen)-1]
    print("\n== RGB 漂移检验 (生成 vs GT) ==")
    for ix in idxs:
        print(f"  frame {ix:4d}: gen={np.round(rgb_mean(gen[ix]),1)}  gt={np.round(rgb_mean(rgb_list[ix]),1)}")
    print(f"\n[{name}] full_PSNR={psnr['full_mean']:.3f}dB "
          f"views={ {k:round(v,2) for k,v in psnr['view_means'].items()} } gen={sec:.1f}s")
    json.dump({"rel":rel,"name":name,"full_mean":psnr["full_mean"],
               "view_means":psnr["view_means"],"gen_seconds":sec},
              open(os.path.join(args.out_dir, f"{name}_v2.json"),"w"), indent=2)
    print(f"输出 -> {args.out_dir}/{name}_v2.mp4 (+ _gt.mp4)")
