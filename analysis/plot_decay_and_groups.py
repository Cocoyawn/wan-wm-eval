import os
import csv
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEWS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def load_all(out_root):
    recs = []
    for f in sorted(glob.glob(os.path.join(out_root, "*", "psnr.json"))):
        p = json.load(open(f))
        rel = p["rel_path"]
        data_type = rel.split("/")[1] if "/" in rel else "unknown"
        recs.append({
            "name": p["name"],
            "rel_path": rel,
            "data_type": data_type,
            "num_frames": p["num_frames"],
            "full_mean": p["full_mean"],
            "view_means": p["view_means"],
            "full_pf": np.array(p["detail"]["full"]["per_frame"], dtype=np.float64),
            "view_pf": {v: np.array(p["detail"][v]["per_frame"], dtype=np.float64) for v in VIEWS},
        })
    return recs


def decay_curve(per_frame_list, max_len=None):
    """把不等长的逐帧 PSNR 对齐到帧索引，按有效计数求均值。inf 视为缺失略过。"""
    if max_len is None:
        max_len = max(len(a) for a in per_frame_list)
    acc = np.zeros(max_len)
    cnt = np.zeros(max_len)
    for a in per_frame_list:
        a = a[:max_len]            # 超长轨迹截断到 max_len
        n = len(a)
        finite = np.isfinite(a)
        acc[:n][finite] += a[finite]
        cnt[:n][finite] += 1
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return mean, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="outputs/eval_tower_of_hanoi_epoch99")
    ap.add_argument("--cap", type=int, default=1500, help="衰减曲线最多画到多少帧(避免极少数超长轨迹拖尾)")
    args = ap.parse_args()

    recs = load_all(args.out_root)
    print(f"载入 {len(recs)} 条")

    cap = args.cap
    full_mean, full_cnt = decay_curve([r["full_pf"] for r in recs], max_len=cap)
    view_curves = {v: decay_curve([r["view_pf"][v] for r in recs], max_len=cap)[0] for v in VIEWS}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(cap)
    ax1.plot(x, full_mean, label="full", color="black", lw=2)
    for v, c in zip(VIEWS, ["#d62728", "#1f77b4", "#2ca02c"]):
        ax1.plot(x, view_curves[v], label=v, alpha=0.8)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"PSNR vs rollout frame index (avg over {len(recs)} val trajectories, epoch-99)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, cap)

    ax2.plot(x, full_cnt, color="gray")
    ax2.set_ylabel("# traj")
    ax2.set_xlabel("frame index")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, cap)
    plt.tight_layout()
    curve_path = os.path.join(args.out_root, "psnr_decay_curve.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"[1] 衰减曲线 -> {curve_path}")
    np.savez(os.path.join(args.out_root, "psnr_decay_curve.npz"),
             frame_idx=x, full=full_mean, count=full_cnt,
             **{v: view_curves[v] for v in VIEWS})

    groups = {}
    for r in recs:
        groups.setdefault(r["data_type"], []).append(r)

    def grp_stats(rs):
        full = np.array([r["full_mean"] for r in rs if np.isfinite(r["full_mean"])])
        out = {
            "count": len(rs),
            "full_mean": float(full.mean()),
            "full_std": float(full.std()),
            "full_min": float(full.min()),
            "full_max": float(full.max()),
        }
        for v in VIEWS:
            vv = np.array([r["view_means"][v] for r in rs if np.isfinite(r["view_means"][v])])
            out[f"{v}_mean"] = float(vv.mean())
        return out

    group_report = {g: grp_stats(rs) for g, rs in groups.items()}
    group_report["ALL"] = grp_stats(recs)

    with open(os.path.join(args.out_root, "group_stats.json"), "w") as f:
        json.dump(group_report, f, indent=2)

    print("\n[2] 分组统计 (整图 PSNR):")
    print(f"  {'group':24s} {'n':>3s} {'mean':>8s} {'std':>6s} {'min':>7s} {'max':>7s}")
    for g, s in group_report.items():
        print(f"  {g:24s} {s['count']:>3d} {s['full_mean']:>8.3f} {s['full_std']:>6.2f} "
              f"{s['full_min']:>7.2f} {s['full_max']:>7.2f}")

    with open(os.path.join(args.out_root, "group_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "count", "full_mean", "full_std", "full_min", "full_max",
                    "cam_high_mean", "cam_left_wrist_mean", "cam_right_wrist_mean"])
        for g, s in group_report.items():
            w.writerow([g, s["count"], round(s["full_mean"], 3), round(s["full_std"], 3),
                        round(s["full_min"], 3), round(s["full_max"], 3),
                        round(s["cam_high_mean"], 3), round(s["cam_left_wrist_mean"], 3),
                        round(s["cam_right_wrist_mean"], 3)])

    fig, ax = plt.subplots(figsize=(9, 5))
    glist = [g for g in group_report if g != "ALL"]
    metrics = ["full_mean"] + [f"{v}_mean" for v in VIEWS]
    labels = ["full"] + VIEWS
    width = 0.8 / len(metrics)
    xpos = np.arange(len(glist))
    for j, (m, lab) in enumerate(zip(metrics, labels)):
        ax.bar(xpos + j * width, [group_report[g][m] for g in glist], width, label=lab)
    ax.set_xticks(xpos + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(glist, rotation=10)
    ax.set_ylabel("mean PSNR (dB)")
    ax.set_title("PSNR by data group (epoch-99)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    bar_path = os.path.join(args.out_root, "group_bar.png")
    plt.savefig(bar_path, dpi=120)
    plt.close()
    print(f"\n[3] 分组柱状图 -> {bar_path}")


if __name__ == "__main__":
    main()
