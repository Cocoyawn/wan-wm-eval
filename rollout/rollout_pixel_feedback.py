import os
os.environ["WAN_ACTION_DIM"] = "14"
os.environ["WAN_CONDITION_FRAMES"] = "9"

import json
import time
import argparse

import torch
import numpy as np
from PIL import Image
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth import save_video
from tqdm import tqdm

# 配置常量（对齐训练脚本）：
#   --height 544 --width 320 --num_frames 57
#   --condition_frames 9 --Ta 48 --To 8 --action_dim 14
GEN_HEIGHT = 544
GEN_WIDTH = 320
CONDITION_FRAMES = 9
PREDICT_FRAMES = 48
STEPS = 5

DATA_ROOT = "/path/to/Challenge-phase1-dataset-rlinf"
CKPT_PATH = "/path/to/ckpt/epoch-99.safetensors"
VAE_PATH = "/path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

# 视角竖直拼接边界（实测：有效 540 行 = 3×180，底部 [540:544] 为黑色 padding）
VIEW_BOUNDS = {
    "cam_high": (0, 180),
    "cam_left_wrist": (180, 360),
    "cam_right_wrist": (360, 540),
}

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt", type=str, default=CKPT_PATH, help="DiT ckpt 路径")
parser.add_argument("--list_file", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "val_list.txt"))
parser.add_argument("--out_root", type=str, default="outputs/eval_rollout_v2")
parser.add_argument("--shard", type=int, default=0, help="本 worker 的分片编号")
parser.add_argument("--num_shards", type=int, default=1, help="总分片数")
parser.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条（调试用）")
parser.add_argument("--save_png", action="store_true", help="是否额外保存逐帧 PNG")
parser.add_argument("--save_gt", action="store_true", help="是否额外保存 GT 视频用于对比")
parser.add_argument("--steps", type=int, default=STEPS, help="去噪步数(覆盖默认5)")
args_cli = parser.parse_args()
STEPS = args_cli.steps   # 用命令行步数覆盖默认

# 世界模型加载
print(f"[{args_cli.device}] 加载模型 ckpt={args_cli.ckpt}")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=args_cli.device,
    model_configs=[
        ModelConfig(path=args_cli.ckpt, offload_device="cpu"),
        ModelConfig(path=VAE_PATH, offload_device="cpu"),
    ],
)
pipe.dit.to(args_cli.device)
pipe.vae.to(args_cli.device)


def load_gt_npy_folder(folder):
    """读取 rgb.npy + actions.npy，返回 (List[PIL.Image] 544x320, actions[T,14])。"""
    rgb = np.load(os.path.join(folder, "rgb.npy"))
    ak = np.load(os.path.join(folder, "actions.npy"))
    if rgb.ndim == 5:
        rgb = rgb[:, 0]  # [T, N, 3, H, W] -> [T, 3, H, W]
    if ak.ndim == 3:
        ak = ak[:, 0]    # [T, N, action_dim] -> [T, action_dim]

    video = []
    for frame in rgb:
        img = frame
        if img.max() <= 1:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        video.append(Image.fromarray(img))
    return video, ak


def save_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"frame_{i:04d}.png"))


def _to_uint8_hwc(img):
    """PIL.Image 或 ndarray -> uint8 HWC ndarray。"""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    return arr


def _psnr_uint8(a, b):
    """两张 uint8 同尺寸图像的 PSNR(dB)。完全相同返回 inf。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def compute_psnr(gen_frames, gt_frames):
    """
    整图 + 分视角逐帧 PSNR。
    gen_frames / gt_frames: List[PIL.Image] 或 ndarray，均为 544x320。
    对齐到 min(len) 帧。视角在原图(544 高)空间按 VIEW_BOUNDS 切分，
    与训练 camera_layout=vertical:cam_high,cam_left_wrist,cam_right_wrist 语义对齐。
    """
    n = min(len(gen_frames), len(gt_frames))
    full_list = []
    view_lists = {k: [] for k in VIEW_BOUNDS}

    for i in range(n):
        g = _to_uint8_hwc(gen_frames[i])
        t = _to_uint8_hwc(gt_frames[i])
        # 尺寸保护：若生成与 GT 尺寸不一致，把生成 resize 到 GT 尺寸再比
        if g.shape[:2] != t.shape[:2]:
            g = np.asarray(
                Image.fromarray(g).resize((t.shape[1], t.shape[0])), dtype=np.uint8
            )
        full_list.append(_psnr_uint8(g, t))
        for name, (lo, hi) in VIEW_BOUNDS.items():
            view_lists[name].append(_psnr_uint8(g[lo:hi], t[lo:hi]))

    def _mean_finite(xs):
        finite = [x for x in xs if np.isfinite(x)]
        return float(np.mean(finite)) if finite else float("inf")

    result = {
        "num_frames": n,
        "full": {"per_frame": full_list, "mean": _mean_finite(full_list)},
    }
    for name in VIEW_BOUNDS:
        result[name] = {
            "per_frame": view_lists[name],
            "mean": _mean_finite(view_lists[name]),
        }
    return result


# 自回归生成（严格输出 action_len 帧）
def build_action_window(actions, start, Ta=48, To=8):
    """
    严格复刻 RLinfDataset(retain_actions=True, action2obs_bias=False) 的 action 窗口构造，
    保证训练/推理 action 语义完全一致。
    返回长度 Ta+To+1 = 57 的 action 窗口：
        [全零(reference)] + actions[start-To+1 : start+Ta+1]
    越界部分：早期 zero-pad；尾部 clamp 到最后一帧。
    """
    dim = actions.shape[1]
    action_s, action_e = start - To + 1, start + Ta + 1   # retain_actions=True 分支

    if action_s < 0:
        # vs<0：pad To 个零 + actions[:Ta]
        pad = np.zeros((To, dim), dtype=actions.dtype)
        act_win = np.concatenate([pad, actions[:Ta]], axis=0)        # 长度 To+Ta=56
    else:
        # 尾部可能越界：clamp 到最后一帧
        idx = np.clip(np.arange(action_s, action_e), 0, len(actions) - 1)
        act_win = actions[idx]                                       # 长度 56
    # 左 pad 1 个全零(reference 帧动作)，与 left_padding_length=1 一致
    act_win = np.concatenate([np.zeros((1, dim), dtype=actions.dtype), act_win], axis=0)
    return act_win                                                   # 长度 57


def generate_sequence(rgb_list, actions, condition_frames=CONDITION_FRAMES,
                      predict_frames=PREDICT_FRAMES, steps=STEPS):
    actions = np.asarray(actions)
    action_len = len(actions)
    window = condition_frames + predict_frames                       # 57
    print(f"动作帧数 = {action_len}")

    num_iters = max(1, (action_len - 1 + predict_frames - 1) // predict_frames)
    print(f"Rolling chunks = {num_iters}")

    generated_frames = []
    input_image = rgb_list[0]                                        # reference 始终是 GT 首帧
    input_image4 = [input_image] * (condition_frames - 1)            # 初始 context

    for i in range(num_iters):
        print(f"\n--- Chunk {i+1}/{num_iters} ---")
        start = i * predict_frames

        act_win = build_action_window(actions, start, Ta=predict_frames, To=condition_frames - 1)
        act_win = torch.from_numpy(np.ascontiguousarray(act_win)).to(
            dtype=torch.bfloat16, device=args_cli.device)

        out_video = pipe(
            seed=0,
            tiled=False,
            input_image=input_image,
            input_image4=input_image4,
            action=act_win,
            height=GEN_HEIGHT,
            width=GEN_WIDTH,
            num_frames=window,
            num_inference_steps=steps,
            cfg_scale=1.0,
        )

        gen_video = out_video[0]

        if len(generated_frames) == 0:
            generated_frames.extend([gen_video[0]] + gen_video[-predict_frames:])
        else:
            generated_frames.extend(gen_video[-predict_frames:])

        # 下一 chunk 的 context = 上一段生成的尾帧(pixel，pipe 内部会 re-encode)
        input_image4 = generated_frames[-(condition_frames - 1):]

    generated_frames = generated_frames[:action_len]
    print(f"最终生成帧数 = {len(generated_frames)}")
    return generated_frames


# 处理单条序列
def process_one_sequence(rel_path, out_root, save_png=False, save_gt=False):
    folder = os.path.join(DATA_ROOT, rel_path)
    print(f"\n=== 处理 {folder} ===")

    rgb_list, actions = load_gt_npy_folder(folder)

    t0 = time.time()
    gen_frames = generate_sequence(rgb_list, actions, steps=args_cli.steps)
    gen_sec = time.time() - t0

    # PSNR：生成 vs GT（GT 截到生成长度）
    psnr = compute_psnr(gen_frames, rgb_list)

    # 输出目录命名：{data_type}__{seed_seg}，避免 failure/success 同名冲突
    parts = rel_path.strip("/").split("/")
    # parts 形如 tower-of-hanoi-game/<data_type>/step_000/seed_xxx_seg_xxx
    data_type = parts[1] if len(parts) > 1 else "data"
    leaf = parts[-1]
    name = f"{data_type}__{leaf}"
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    # 保存 infer 视频
    video_path = os.path.join(out_dir, "video.mp4")
    save_video(gen_frames, video_path, fps=30, quality=5)

    if save_gt:
        save_video(rgb_list[:len(gen_frames)], os.path.join(out_dir, "gt.mp4"), fps=30, quality=5)
    if save_png:
        save_frames(gen_frames, os.path.join(out_dir, "images"))

    # 保存 PSNR
    psnr_out = {
        "rel_path": rel_path,
        "name": name,
        "gen_seconds": gen_sec,
        "num_frames": psnr["num_frames"],
        "full_mean": psnr["full"]["mean"],
        "view_means": {k: psnr[k]["mean"] for k in VIEW_BOUNDS},
        "detail": psnr,
    }
    with open(os.path.join(out_dir, "psnr.json"), "w") as f:
        json.dump(psnr_out, f, indent=2)

    print(f"[{name}] full_PSNR={psnr['full']['mean']:.3f}dB "
          f"views={ {k: round(psnr[k]['mean'],2) for k in VIEW_BOUNDS} } "
          f"gen={gen_sec:.1f}s")
    return psnr_out


# 批处理（支持分片）
if __name__ == "__main__":
    with open(args_cli.list_file) as f:
        all_items = [ln.strip() for ln in f if ln.strip()]
    if args_cli.limit > 0:
        all_items = all_items[:args_cli.limit]

    # 分片：idx % num_shards == shard
    items = [p for i, p in enumerate(all_items)
             if i % args_cli.num_shards == args_cli.shard]
    print(f"[{args_cli.device}] shard {args_cli.shard}/{args_cli.num_shards} "
          f"处理 {len(items)}/{len(all_items)} 条")

    os.makedirs(args_cli.out_root, exist_ok=True)
    results = []
    t_start = time.time()
    for rel_path in items:
        try:
            r = process_one_sequence(rel_path, args_cli.out_root,
                                     save_png=args_cli.save_png, save_gt=args_cli.save_gt)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"!!! 失败 {rel_path}: {e}")
            traceback.print_exc()
            results.append({"rel_path": rel_path, "error": str(e)})

    # 落盘本分片结果（汇总由外部脚本合并所有分片）
    shard_out = os.path.join(args_cli.out_root, f"_shard_{args_cli.shard}_of_{args_cli.num_shards}.json")
    with open(shard_out, "w") as f:
        json.dump({
            "device": args_cli.device,
            "shard": args_cli.shard,
            "num_shards": args_cli.num_shards,
            "total_seconds": time.time() - t_start,
            "results": results,
        }, f, indent=2)
    print(f"[{args_cli.device}] shard 完成，用时 {time.time()-t_start:.1f}s -> {shard_out}")
