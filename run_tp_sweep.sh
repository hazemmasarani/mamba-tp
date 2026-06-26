#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="log"
mkdir -p "$LOG_DIR"

IN_SCRIPT="input/gen_input.py"
RUN_SCRIPT="run_cache.py"

# batch sizes
BATCH_SIZES=(1 2 4 8 16 32 64)

# compute max seq len per batch size
get_max_seq() {
  local bs=$1
  echo $((1024 / bs))
}

echo "Starting TP benchmark sweep..."
echo "Logs: $LOG_DIR"

for bs in "${BATCH_SIZES[@]}"; do

  max_seq=$(get_max_seq "$bs")

  # enforce minimum seq length = 4
  if [ "$max_seq" -lt 4 ]; then
    max_seq=4
  fi

  echo "========================================="
  echo "Batch size: $bs | Seq_len: 4 → $max_seq"
  echo "========================================="

  seq=4
  while [ "$seq" -le "$max_seq" ]; do

    echo "[RUN] bs=$bs seq=$seq"

    log_file="$LOG_DIR/tp_bs${bs}_seq${seq}.log"

    python "$IN_SCRIPT" \
      -batch_size "$bs" \
      -seq_len "$seq" \
      2>&1 | tee "$log_file"
    
    python "$RUN_SCRIPT"

    # exponential stepping for speed (change to seq=$((seq+4)) if you want linear)
    seq=$((seq * 2))

  done

done

echo "Done."