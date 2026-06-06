"""
纯 latent 空间自回归 rollout —— 修正 AR 的 VAE round-trip 问题。
AR 用 decode 后的 pixel 当 condition, 每 chunk 一次 decode->encode, VAE 重建噪声累积;
改法: chunk 间直接传 latent, 不经过 pixel。

monkey-patch:
  1. ImageEmbedderFused: pipe._cond_latent 已设时直接用它作 condition latent, 跳过 pixel encode
  2. vae.decode: decode 前把完整 latent 存到 pipe._last_latent, 供下一 chunk 取尾部 3 帧作 condition
  3. 每 chunk 只保存 predict 部分 latent, 最后拼接一次性 decode
"""
import os
os.environ["WAN_ACTION_DIM"] = "14"
os.environ["WAN_CONDITION_FRAMES"] = "9"

import json
import time
import types
import argparse

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import diffsynth.pipelines.wan_video_new as wvn
from diffsynth import save_video

GEN_HEIGHT, GEN_WIDTH = 544, 320
CONDITION_FRAMES, PREDICT_FRAMES = 9, 48
COND_LATENT = (CONDITION_FRAMES - 1) // 4 + 1            # 3
WINDOW = CONDITION_FRAMES + PREDICT_FRAMES              # 57
VAE_PATH = "/path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
DATA_ROOT = "/path/to/Challenge-phase1-dataset-rlinf"
VIEW_BOUNDS = {"cam_high": (0, 180), "cam_left_wrist": (180, 360), "cam_right_wrist": (360, 540)}

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--list_file", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "val_list.txt"))
parser.add_argument("--out_root", type=str, default="outputs/eval_rollout_latent")
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--save_gt", action="store_true")
args = parser.parse_args()

# ---- 加载 pipe ----
print(f"[{args.device}] 加载 ckpt={args.ckpt}")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device=args.device,
    model_configs=[ModelConfig(path=args.ckpt, offload_device="cpu"),
                   ModelConfig(path=VAE_PATH, offload_device="cpu")])
pipe.dit.to(args.device)
pipe.vae.to(args.device)

# ================= monkey-patch =================
# 状态: pipe._cond_latent (外部注入的 condition latent, [B,C,T0,Hl,Wl] 或 None)
#       pipe._last_latent (去噪后完整 latent, decode 前截获)
pipe._cond_latent = None
pipe._last_latent = None

# patch 1: ImageEmbedderFused — 优先用注入的 latent condition, 跳过 pixel encode
_orig_imgfused_process = None
for u in pipe.units:
    if u.__class__.__name__ == "WanVideoUnit_ImageEmbedderFused":
        _img_unit = u
        _orig_imgfused_process = u.__class__.process
        break

def patched_imgfused_process(self, pipe, input_image, latents, height, width,
                             tiled, tile_size, tile_stride, input_image4):
    if getattr(pipe, "_cond_latent", None) is not None:
        # 直接用注入的 condition latent, 不做任何 VAE encode
        z = pipe._cond_latent.to(dtype=latents.dtype, device=latents.device)
        T0 = z.shape[2]
        latents[:, :, :T0] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}
    # 否则走原逻辑(首 chunk 用 GT 首帧 encode)
    return _orig_imgfused_process(self, pipe, input_image, latents, height, width,
                                  tiled, tile_size, tile_stride, input_image4)

_img_unit.__class__.process = patched_imgfused_process

# patch 2: vae.decode — decode 前截获 latent
_orig_decode = pipe.vae.decode
def patched_decode(hidden_states, *a, **k):
    pipe._last_latent = hidden_states.detach().clone()
    return _orig_decode(hidden_states, *a, **k)
pipe.vae.decode = patched_decode
# ===============================================


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
    return win   # 57


@torch.no_grad()
def latent_rollout(rgb_list, actions, steps):
    actions = np.asarray(actions)
    action_len = len(actions)
    num_iters = max(1, (action_len - 1 + PREDICT_FRAMES - 1) // PREDICT_FRAMES)
    print(f"动作帧数={action_len} chunks={num_iters} (latent-space AR)")

    first_frame = rgb_list[0]
    all_pred_latents = []   # 每 chunk 的 predict latent (跳过 condition 部分)
    first_chunk_full = None
    gt_boot_latent = None   # 轨迹首帧的"开机帧"latent (slot0), 整条 rollout 复用

    for i in range(num_iters):
        print(f"--- Chunk {i+1}/{num_iters} ---")
        start = i * PREDICT_FRAMES
        act = build_action_window(actions, start, Ta=PREDICT_FRAMES, To=CONDITION_FRAMES-1)
        act = torch.from_numpy(np.ascontiguousarray(act)).to(dtype=torch.bfloat16, device=args.device)

        if i == 0:
            pipe._cond_latent = None       # 首 chunk: 用 GT 首帧 encode (原逻辑)
            _ = pipe(seed=0, tiled=False, input_image=first_frame,
                     input_image4=[first_frame]*(CONDITION_FRAMES-1),
                     action=act, height=GEN_HEIGHT, width=GEN_WIDTH, num_frames=WINDOW,
                     num_inference_steps=steps, cfg_scale=1.0)
            full = pipe._last_latent        # [B,C,15,Hl,Wl]
            first_chunk_full = full
            gt_boot_latent = full[:, :, 0:1].clone()   # 钉死轨迹首帧的开机帧 latent
            all_pred_latents.append(full[:, :, COND_LATENT:])   # 去掉前3 condition latent
            prev_latent = full
        else:
            # condition: slot0 = 轨迹首帧开机帧(钉死), slot1,2 = 上一段尾部2帧 (纯 latent)
            pipe._cond_latent = torch.cat(
                [gt_boot_latent, prev_latent[:, :, -(COND_LATENT-1):]], dim=2).clone()
            _ = pipe(seed=0, tiled=False, input_image=first_frame,   # input_image 仅占位
                     input_image4=[first_frame]*(CONDITION_FRAMES-1),
                     action=act, height=GEN_HEIGHT, width=GEN_WIDTH, num_frames=WINDOW,
                     num_inference_steps=steps, cfg_scale=1.0)
            full = pipe._last_latent
            all_pred_latents.append(full[:, :, COND_LATENT:])
            prev_latent = full

    pipe._cond_latent = None
    # 拼接: 首chunk全部(含首帧) + 各chunk predict
    full_latent = torch.cat([first_chunk_full[:, :, :COND_LATENT]] + all_pred_latents, dim=2)
    # 一次性 decode
    pipe.load_models_to_device(["vae"])
    video = _orig_decode(full_latent, device=args.device, tiled=False)
    frames = pipe.vae_output_to_video(video)[0]
    frames = frames[:action_len]
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
        if g.shape[:2]!=t.shape[:2]:
            g=np.asarray(Image.fromarray(g).resize((t.shape[1],t.shape[0])),dtype=np.uint8)
        full.append(_psnr(g,t))
        for v,(lo,hi) in VIEW_BOUNDS.items(): views[v].append(_psnr(g[lo:hi],t[lo:hi]))
    mean=lambda xs:float(np.mean([x for x in xs if np.isfinite(x)])) if any(np.isfinite(xs)) else float('inf')
    return {"num_frames":n, "full_mean":mean(full), "view_means":{v:mean(views[v]) for v in VIEW_BOUNDS},
            "full_pf":full, "view_pf":views}


def process_one(rel, out_root, save_gt):
    folder=os.path.join(DATA_ROOT, rel)
    print(f"\n=== {folder} ===")
    rgb_list, actions = load_gt(folder)
    t0=time.time()
    gen = latent_rollout(rgb_list, actions, args.steps)
    sec=time.time()-t0
    psnr=compute_psnr(gen, rgb_list)
    parts=rel.strip("/").split("/"); name=f"{parts[1]}__{parts[-1]}"
    od=os.path.join(out_root,name); os.makedirs(od,exist_ok=True)
    save_video(gen, os.path.join(od,"video.mp4"), fps=30, quality=5)
    if save_gt: save_video(rgb_list[:len(gen)], os.path.join(od,"gt.mp4"), fps=30, quality=5)
    rec={"rel_path":rel,"name":name,"gen_seconds":sec,"num_frames":psnr["num_frames"],
         "full_mean":psnr["full_mean"],"view_means":psnr["view_means"],
         "detail":{"num_frames":psnr["num_frames"],
                   "full":{"per_frame":psnr["full_pf"],"mean":psnr["full_mean"]},
                   **{v:{"per_frame":psnr["view_pf"][v],"mean":psnr["view_means"][v]} for v in VIEW_BOUNDS}}}
    json.dump(rec, open(os.path.join(od,"psnr.json"),"w"), indent=2)
    print(f"[{name}] full_PSNR={psnr['full_mean']:.3f}dB views={ {k:round(v,2) for k,v in psnr['view_means'].items()} } gen={sec:.1f}s")
    return rec


if __name__ == "__main__":
    items=[l.strip() for l in open(args.list_file) if l.strip()]
    if args.limit: items=items[:args.limit]
    items=[p for i,p in enumerate(items) if i%args.num_shards==args.shard]
    os.makedirs(args.out_root,exist_ok=True)
    res=[]; t0=time.time()
    for rel in items:
        try: res.append(process_one(rel,args.out_root,args.save_gt))
        except Exception as e:
            import traceback; print(f"!!! {rel}: {e}"); traceback.print_exc()
            res.append({"rel_path":rel,"error":str(e)})
    out=os.path.join(args.out_root,f"_shard_{args.shard}_of_{args.num_shards}.json")
    json.dump({"device":args.device,"ckpt":args.ckpt,"shard":args.shard,"num_shards":args.num_shards,
               "total_seconds":time.time()-t0,"results":res}, open(out,"w"), indent=2)
    print(f"[{args.device}] 完成 -> {out} 用时{time.time()-t0:.1f}s")
