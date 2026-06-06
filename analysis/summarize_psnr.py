import os
import csv
import json
import glob
import argparse

import numpy as np

VIEWS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="outputs/eval_tower_of_hanoi_epoch99")
    args = ap.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.out_root, "_shard_*_of_*.json")))
    if not shard_files:
        raise FileNotFoundError(f"在 {args.out_root} 没找到分片结果 _shard_*.json")

    rows = []
    max_total_seconds = 0.0
    for sf in shard_files:
        data = json.load(open(sf))
        max_total_seconds = max(max_total_seconds, data.get("total_seconds", 0.0))
        for r in data["results"]:
            if "error" in r:
                rows.append({"name": r.get("rel_path", "?"), "error": r["error"]})
                continue
            row = {
                "name": r["name"],
                "rel_path": r["rel_path"],
                "num_frames": r["num_frames"],
                "full_mean": r["full_mean"],
                "gen_seconds": r.get("gen_seconds", float("nan")),
            }
            for v in VIEWS:
                row[v] = r["view_means"].get(v, float("nan"))
            rows.append(row)

    ok_rows = [r for r in rows if "error" not in r]
    err_rows = [r for r in rows if "error" in r]

    def _mean(key):
        vals = [r[key] for r in ok_rows if np.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    overall = {
        "num_evaluated": len(ok_rows),
        "num_errors": len(err_rows),
        "wallclock_slowest_shard_seconds": max_total_seconds,
        "full_mean_psnr": _mean("full_mean"),
        "view_mean_psnr": {v: _mean(v) for v in VIEWS},
    }

    # summary.json
    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump({"overall": overall, "per_sequence": rows}, f, indent=2)

    # summary.csv
    csv_path = os.path.join(args.out_root, "summary.csv")
    fields = ["name", "num_frames", "full_mean"] + VIEWS + ["gen_seconds", "rel_path"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(ok_rows, key=lambda x: x["name"]):
            w.writerow(r)

    # 控制台报告
    print("=" * 60)
    print(f"评估完成: {overall['num_evaluated']} 条 (错误 {overall['num_errors']} 条)")
    print(f"整图平均 PSNR : {overall['full_mean_psnr']:.3f} dB")
    for v in VIEWS:
        print(f"  {v:16s}: {overall['view_mean_psnr'][v]:.3f} dB")
    print(f"最慢分片墙钟  : {max_total_seconds/60:.1f} min")
    print(f"明细 -> {csv_path}")
    if err_rows:
        print("!!! 失败条目:")
        for r in err_rows:
            print("   ", r["name"], "-", r["error"])
    print("=" * 60)


if __name__ == "__main__":
    main()
