#!/bin/bash
# run_experiments.sh - the ablation sweep behind the experiments write-up.
#
# Eleven runs. Every run: 1000 iterations, 16,384 characters per iteration (batch x context
# is held constant), eval every 100 iterations on 20 batches. Only one knob
# changes per run relative to the baseline (3.2M params, context 256, lr 1e-3).
# Sequential on purpose: two runs on one GPU would corrupt the timings.
set -e
cd "$(dirname "$0")"
mkdir -p runs
COMMON="--max_iters 1000 --eval_interval 100 --eval_iters 20 --warmup_iters 100"
SMALL="--n_layer 4 --n_head 4 --n_embd 256"

run() {  # run <name> <extra args...>
  name=$1; shift
  # Let Metal's compiler service settle between processes. Launching many
  # short-lived GPU processes back to back can exhaust it, which aborts every
  # later run instantly with "Unable to reach MTLCompilerService".
  sleep 20
  echo "=== START $name: $*   ($(date +%H:%M:%S))"
  python3 -u train.py $COMMON "$@" --out_dir "runs/$name" > "runs/$name.log" 2>&1
  grep -E "^--- eval @ 1000|^done" "runs/$name.log"
  echo "=== END $name ($(date +%H:%M:%S))"
}

# Axis 1: chunk size, batch adjusted so characters/iter is constant.
# Seven sizes: four points suggested a plateau, and the three in between
# revealed the actual minimum.
run ctx0016   $SMALL --block_size 16   --batch_size 1024
run ctx0032   $SMALL --block_size 32   --batch_size 512
run ctx0064   $SMALL --block_size 64   --batch_size 256
run ctx0128   $SMALL --block_size 128  --batch_size 128
run ctx0256   $SMALL --block_size 256  --batch_size 64      # baseline; shared with axes 2 and 3
# Axis 2: model size at context 256
run size0.4M  --n_layer 2 --n_head 2 --n_embd 128
# Axis 3: learning rate on the baseline model
run lr3e-4    $SMALL --learning_rate 3e-4 --min_lr 3e-5
run lr3e-3    $SMALL --learning_rate 3e-3 --min_lr 3e-4
# The slow runs last
run ctx0512   $SMALL --block_size 512  --batch_size 32
run ctx1024   $SMALL --block_size 1024 --batch_size 16
run size10.6M --n_layer 6 --n_head 6 --n_embd 384
echo "=== ALL DONE ($(date +%H:%M:%S))"
