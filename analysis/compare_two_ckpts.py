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


def load(out_root):
    recs = {}
    for f in sorted(glob.glob(os.path.join(out_root, "*", "psnr.json"))):
        p = json.load(open(f))
        recs[p["name"]] = {
            "full_mean": p["full_mean"],
            "view_means": p["view_means"],
            "num_frames": p["num_frames"],
            "data_type": p["rel_path"].split("/")[1],
            "full_pf": np.array(p["detail"]["full"]["per_frame"], dtype=np.float64),
        }
    return recs


def decay(per_frame_list, cap):
    acc = np.zeros(cap); cnt = np.zeros(cap)
    for a in per_frame_list:
        a = a[:cap]; n = len(a); fin = np.isfinite(a)
        acc[:n][fin] += a[fin]; cnt[:n][fin] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs/eval_tower_of_hanoi_epoch99")
    ap.add_argument("--a_name", default="epoch-99")
    ap.add_argument("--b", default="outputs/eval_step42000")
    ap.add_argument("--b_name", default="step-42000")
    ap.add_argument("--out", default="outputs/compare_epoch99_vs_step42000")
    ap.add_argument("--cap", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A = load(args.a); B = load(args.b)
    common = sorted(set(A) & set(B))
    print(f"{args.a_name}: {len(A)} 条, {args.b_name}: {len(B)} 条, 公共 {len(common)} 条")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(args.cap)
    ax.plot(x, decay([A[k]["full_pf"] for k in A], args.cap), label=f"{args.a_name} full", color="#1f77b4", lw=2)
    ax.plot(x, decay([B[k]["full_pf"] for k in B], args.cap), label=f"{args.b_name} full", color="#d62728", lw=2)
    ax.set_xlabel("rollout frame index"); ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR decay: {args.a_name} vs {args.b_name} (avg over val trajectories)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, args.cap)
    plt.tight_layout(); p1 = os.path.join(args.out, "decay_overlay.png")
    plt.savefig(p1, dpi=120); plt.close()
    print(f"[1] 叠加衰减曲线 -> {p1}")

    def group_mean(recs, dt=None):
        vals = [r["full_mean"] for r in recs.values()
                if (dt is None or r["data_type"] == dt) and np.isfinite(r["full_mean"])]
        return float(np.mean(vals)) if vals else float("nan")

    groups = ["ALL", "failure-data", "success-and-hil-data"]
    a_vals = [group_mean(A, None if g == "ALL" else g) for g in groups]
    b_vals = [group_mean(B, None if g == "ALL" else g) for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    xp = np.arange(len(groups)); w = 0.35
    b1 = ax.bar(xp - w/2, a_vals, w, label=args.a_name, color="#1f77b4")
    b2 = ax.bar(xp + w/2, b_vals, w, label=args.b_name, color="#d62728")
    ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(xp); ax.set_xticklabels(groups)
    ax.set_ylabel("mean full PSNR (dB)"); ax.set_title("PSNR by group: two ckpts")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); p2 = os.path.join(args.out, "group_compare.png")
    plt.savefig(p2, dpi=120); plt.close()
    print(f"[2] 分组对比柱状图 -> {p2}")

    # 配对差异：同一条轨迹 B-A
    diffs = []
    for k in common:
        d = B[k]["full_mean"] - A[k]["full_mean"]
        diffs.append((k, A[k]["full_mean"], B[k]["full_mean"], d, A[k]["data_type"]))
    diffs.sort(key=lambda r: r[3])
    darr = np.array([r[3] for r in diffs])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(darr, bins=25, color="#7f7f7f", edgecolor="black")
    ax.axvline(0, color="black", ls="--")
    ax.axvline(float(np.mean(darr)), color="red", lw=2, label=f"mean Δ={np.mean(darr):.3f}dB")
    ax.set_xlabel(f"per-trajectory PSNR diff ({args.b_name} − {args.a_name}) dB")
    ax.set_ylabel("# trajectories"); ax.set_title("Paired PSNR difference distribution")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); p3 = os.path.join(args.out, "paired_diff_hist.png")
    plt.savefig(p3, dpi=120); plt.close()
    print(f"[3] 配对差异直方图 -> {p3}")

    with open(os.path.join(args.out, "paired_diff.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", f"{args.a_name}", f"{args.b_name}", "diff_B_minus_A", "data_type"])
        for r in diffs:
            w.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3), r[4]])

    b_better = int((darr > 0).sum())
    summary = {
        "a_name": args.a_name, "b_name": args.b_name,
        "a_full_mean": group_mean(A), "b_full_mean": group_mean(B),
        "a_views": {v: float(np.nanmean([A[k]["view_means"][v] for k in A])) for v in VIEWS},
        "b_views": {v: float(np.nanmean([B[k]["view_means"][v] for k in B])) for v in VIEWS},
        "paired_n": len(common),
        "mean_diff_B_minus_A": float(np.mean(darr)),
        "b_better_count": b_better, "a_better_count": len(common) - b_better,
        "biggest_b_win": diffs[-1][:4], "biggest_a_win": diffs[0][:4],
    }
    json.dump(summary, open(os.path.join(args.out, "compare_summary.json"), "w"), indent=2)

    print("\n===== 对比汇总 =====")
    print(f"整图均值: {args.a_name} {summary['a_full_mean']:.3f}  vs  {args.b_name} {summary['b_full_mean']:.3f}  "
          f"(Δ {summary['b_full_mean']-summary['a_full_mean']:+.3f})")
    for v in VIEWS:
        print(f"  {v:16s}: {summary['a_views'][v]:.3f}  vs  {summary['b_views'][v]:.3f}")
    print(f"配对比较 {len(common)} 条: {args.b_name} 更好 {b_better} 条, {args.a_name} 更好 {len(common)-b_better} 条")
    print(f"平均逐条差 (B−A): {np.mean(darr):+.3f} dB")
    print(f"{args.b_name} 最大胜出: {diffs[-1][0]} ({diffs[-1][3]:+.2f})")
    print(f"{args.a_name} 最大胜出: {diffs[0][0]} ({diffs[0][3]:+.2f})")


if __name__ == "__main__":
    main()
