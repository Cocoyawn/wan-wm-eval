import os
import json
import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

DATA_ROOT = "/path/to/Challenge-phase1-dataset-rlinf"
GAP = 8  # 中间分隔条像素宽


def load_gt(rel_path, n_limit=None):
    rgb = np.load(os.path.join(DATA_ROOT, rel_path, "rgb.npy"), mmap_mode="r")
    T = rgb.shape[0] if n_limit is None else min(n_limit, rgb.shape[0])
    frames = []
    for i in range(T):
        f = np.array(rgb[i, 0])             # [3,544,320] uint8
        f = np.transpose(f, (1, 2, 0))      # HWC
        frames.append(f)
    return frames


def load_gen(out_dir):
    rd = imageio.get_reader(os.path.join(out_dir, "video.mp4"))
    frames = [np.asarray(f)[:, :, :3] for f in rd]
    rd.close()
    return frames


def label(img, text):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, pil.width, 16], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(pil)


def make_pair(rel_path, out_dir, save_path, psnr_mean, stride=1):
    gt = load_gt(rel_path)
    gen = load_gen(out_dir)
    n = min(len(gt), len(gen))
    H = gt[0].shape[0]
    sep = np.full((H, GAP, 3), 128, dtype=np.uint8)

    writer = imageio.get_writer(save_path, fps=30, quality=5,
                                macro_block_size=None)
    for i in range(0, n, stride):
        g = gt[i]
        p = gen[i]
        if p.shape[:2] != g.shape[:2]:
            p = np.asarray(Image.fromarray(p).resize((g.shape[1], g.shape[0])))
        g = label(g, "GT")
        p = label(p, f"GEN {psnr_mean:.1f}dB")
        frame = np.concatenate([g, sep, p], axis=1)
        writer.append_data(frame)
    writer.close()
    print(f"  -> {save_path}  ({n} frames, stride {stride})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="outputs/eval_tower_of_hanoi_epoch99")
    ap.add_argument("--max_frames", type=int, default=900,
                    help="超过则抽帧，使输出帧数不超过该值")
    ap.add_argument("--items", type=str, nargs="*", default=None,
                    help="指定 name 列表；缺省自动选 best/median/worst + 两组中位")
    args = ap.parse_args()

    recs = {}
    for d in sorted(os.listdir(args.out_root)):
        pj = os.path.join(args.out_root, d, "psnr.json")
        if os.path.isfile(pj):
            p = json.load(open(pj))
            recs[p["name"]] = p

    if args.items:
        chosen = args.items
    else:
        srt = sorted(recs.values(), key=lambda x: x["full_mean"])
        fa = [r for r in srt if r["rel_path"].split("/")[1] == "failure-data"]
        su = [r for r in srt if r["rel_path"].split("/")[1] == "success-and-hil-data"]
        picks = {
            "BEST": srt[-1], "MEDIAN": srt[len(srt) // 2], "WORST": srt[0],
            "FAIL_med": fa[len(fa) // 2], "SUCC_med": su[len(su) // 2],
        }
        chosen = []
        seen = set()
        for tag, r in picks.items():
            if r["name"] not in seen:
                chosen.append(r["name"]); seen.add(r["name"])

    cmp_dir = os.path.join(args.out_root, "compare_videos")
    os.makedirs(cmp_dir, exist_ok=True)
    for name in chosen:
        if name not in recs:
            print(f"!! 跳过未知 {name}"); continue
        p = recs[name]
        n = p["num_frames"]
        stride = max(1, (n + args.max_frames - 1) // args.max_frames)
        print(f"[{name}] {p['full_mean']:.2f}dB {n}f")
        make_pair(p["rel_path"], os.path.join(args.out_root, name),
                  os.path.join(cmp_dir, f"{name}_cmp.mp4"),
                  p["full_mean"], stride=stride)
    print(f"\n对比视频目录: {cmp_dir}")


if __name__ == "__main__":
    main()
