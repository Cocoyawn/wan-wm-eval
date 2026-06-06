#!/bin/bash
# 4 卡并行 rollout 评估 tower-of-hanoi val-data（86 条）
# 用法: bash run_4gpu.sh <SCRIPT> [CKPT] [STEPS] [OUT_ROOT]
#   SCRIPT   : rollout/rollout_pixel_feedback.py | rollout/rollout_latent.py | ...
#   CKPT     : DiT ckpt 路径
#   STEPS    : 去噪步数(默认5)
#   OUT_ROOT : 输出目录
# 例:
#   export PYTHONPATH=/path/to/diffsynth-studio-rlinf:$PYTHONPATH
#   bash run_4gpu.sh rollout/rollout_pixel_feedback.py /path/step-42000.safetensors 5 outputs/pixel_full
set -e

cd "$(dirname "$0")"

SCRIPT="${1:?需要指定 rollout 脚本, 如 rollout/rollout_pixel_feedback.py}"
CKPT="${2:?需要指定 ckpt 路径}"
STEPS="${3:-5}"
OUT_ROOT="${4:-outputs/rollout_full}"
NUM_SHARDS=4
mkdir -p "$OUT_ROOT" logs

echo "[launch] 4 卡并行 script=$SCRIPT ckpt=$CKPT steps=$STEPS -> $OUT_ROOT"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" \
    --device cuda:0 \
    --ckpt "$CKPT" \
    --steps "$STEPS" \
    --shard $i --num_shards $NUM_SHARDS \
    --out_root "$OUT_ROOT" \
    --save_gt \
    > "logs/shard_${i}.log" 2>&1 &
  echo "  shard $i -> GPU $i (pid $!)"
done

wait
echo "[done] 全部分片完成 -> $OUT_ROOT"
echo "[next] python analysis/summarize_psnr.py --out_root $OUT_ROOT"
