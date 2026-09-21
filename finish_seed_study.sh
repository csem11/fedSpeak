#!/bin/bash
# finish_seed_study.sh - the eight runs that a Metal compiler failure aborted,
# plus the fair scoring across seeds. About 45 minutes on an idle machine.
#
# Needs a healthy GPU. Check first:
#   python3 -c "import torch; x=torch.randn(64,64,device='mps'); print(float((x@x).sum()))"
# If that aborts with "Unable to reach MTLCompilerService", reboot first.
set -u
cd "$(dirname "$0")"
COMMON="--max_iters 1000 --eval_interval 100 --eval_iters 20 --warmup_iters 100"
SMALL="--n_layer 4 --n_head 4 --n_embd 256"
run() {
  name=$1; shift
  sleep 20                       # let Metal's compiler service settle between processes
  echo "=== START $name ($(date +%H:%M:%S))"
  python3 -u train.py $COMMON "$@" --out_dir "runs/$name" > "runs/$name.log" 2>&1
  grep -E "^--- eval @ 1000" "runs/$name.log" || echo "    FAILED - see runs/$name.log"
  echo "=== END $name ($(date +%H:%M:%S))"
}

# The two learning-rate re-runs that did not get their clean timing
run lr3e-4 $SMALL --learning_rate 3e-4 --min_lr 3e-5
run lr3e-3 $SMALL --learning_rate 3e-3 --min_lr 3e-4

# Two extra seeds either side of the chunk-size minimum
for seed in 1338 1339; do
  run ctx0064_s$seed $SMALL --block_size 64  --batch_size 256 --seed $seed
  run ctx0128_s$seed $SMALL --block_size 128 --batch_size 128 --seed $seed
  run ctx0256_s$seed $SMALL --block_size 256 --batch_size 64  --seed $seed
done

sleep 20
echo "=== fair scoring across all nine ($(date +%H:%M:%S))"
python3 -u eval_common_context.py \
  --runs ctx0064,ctx0064_s1338,ctx0064_s1339,ctx0128,ctx0128_s1338,ctx0128_s1339,ctx0256,ctx0256_s1338,ctx0256_s1339 \
  --out results/common_context_seeds.csv
echo "=== ALL DONE ($(date +%H:%M:%S))"
