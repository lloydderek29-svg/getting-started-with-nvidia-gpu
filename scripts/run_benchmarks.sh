#!/usr/bin/env bash
# scripts/run_benchmarks.sh
#
# Runs the benchmark programs from the book and saves every result
# in one timestamped folder, together with the GPU report and the
# environment snapshot.
#
# Usage: bash scripts/run_benchmarks.sh [--suite quick|python|all]
#                                       [--out DIR]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

BIN="${BIN_DIR:-build/release/bin}"
PYTHON="${PYTHON:-python}"
SUITE="quick"
OUT="results/bench/$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_benchmarks.sh [options]

Options:
  --suite NAME   quick   CUDA C++ benchmarks only (default)
                 python  quick suite plus lightweight Python benchmarks
                 all     python suite plus model based benchmarks
                         (downloads a model and needs the Chapter 6
                         and Chapter 9 artifacts)
  --out DIR      Folder for results
                 (default: results/bench/TIMESTAMP)
  -h, --help     Show this help

Environment variables:
  BIN_DIR        Folder with the built programs
                 (default: build/release/bin)
  PYTHON         Python interpreter to use (default: python)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --suite|--out)
      if [ $# -lt 2 ]; then
        echo "Option $1 needs a value" >&2
        exit 2
      fi
      if [ "$1" = "--suite" ]; then SUITE="$2"; else OUT="$2"; fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SUITE" in
  quick|python|all) ;;
  *)
    echo "Unknown suite: $SUITE" >&2
    exit 2
    ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1 \
   || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No NVIDIA GPU found. Run 'make doctor' to diagnose." >&2
  exit 1
fi

mkdir -p "$OUT"

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
NAMES=()
STATUS=()

record() {
  NAMES+=("$1")
  STATUS+=("$2")
}

skip() {
  printf '%-28s SKIP (%s)\n' "$1" "$2"
  record "$1" SKIP
}

run() {
  local name="$1"
  shift
  printf '%-28s ' "$name"
  if "$@" >"$OUT/$name.log" 2>&1; then
    echo "PASS"
    record "$name" PASS
  else
    echo "FAIL (see $OUT/$name.log)"
    record "$name" FAIL
  fi
}

run_bin() {
  local exe="$1"
  shift
  if [ -x "$BIN/$exe" ]; then
    run "$exe" "$BIN/$exe" "$@"
  else
    skip "$exe" "not built, run: make build"
  fi
}

run_py() {
  local name="$1" script="$2"
  shift 2
  if [ -f "$script" ]; then
    run "$name" "$PYTHON" "$script" "$@"
  else
    skip "$name" "missing $script"
  fi
}

run_hist() {
  local exe="$BIN/hist_bench"
  local report="chapters/ch04_profiling/python/perf_report.py"
  if [ ! -x "$exe" ]; then
    skip hist_bench "not built, run: make build"
    return
  fi
  printf '%-28s ' hist_bench
  if "$exe" >"$OUT/hist.csv" 2>"$OUT/hist_bench.log"; then
    echo "PASS"
    record hist_bench PASS
    if [ -f "$report" ]; then
      "$PYTHON" "$report" "$OUT/hist.csv" \
        >"$OUT/hist_table.md" 2>>"$OUT/hist_bench.log" || true
    fi
  else
    echo "FAIL (see $OUT/hist_bench.log)"
    record hist_bench FAIL
  fi
}

# ---------------------------------------------------------------
# Record the machine before measuring anything
# ---------------------------------------------------------------
echo "Results folder: $OUT"
echo "Suite: $SUITE"
echo

busy=$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader 2>/dev/null | wc -l)
if [ "$busy" -gt 0 ]; then
  echo "Warning: $busy other process(es) are using the GPU."
  echo "Their load will distort the measurements."
  echo
fi

nvidia-smi >"$OUT/nvidia-smi.txt" 2>&1

"$PYTHON" -c \
  "from gpubook.env import save_snapshot; save_snapshot('$OUT/env.json')" \
  >"$OUT/env.log" 2>&1 \
  || echo "Environment snapshot failed, see $OUT/env.log"

if [ -x "$BIN/runtime_report" ]; then
  "$PYTHON" -m gpubook.report --runtime "$BIN/runtime_report" \
    --json "$OUT/gpu_report.json" >"$OUT/gpu_report.txt" 2>&1 \
    || echo "GPU report failed, see $OUT/gpu_report.txt"
else
  "$PYTHON" -m gpubook.report --json "$OUT/gpu_report.json" \
    >"$OUT/gpu_report.txt" 2>&1 \
    || echo "GPU report failed, see $OUT/gpu_report.txt"
fi

# ---------------------------------------------------------------
# Quick suite: CUDA C++ benchmarks (Chapters 1, 3, and 4)
# ---------------------------------------------------------------
echo "CUDA C++ benchmarks"
run_bin bandwidth_test
run_bin transfer_test
run_bin blocksize_sweep
run_bin memory_models
run_bin reduce_sum
run_bin matmul_tiled
run_bin streams_overlap
run_bin graph_demo
run_bin baseline_bench
run_hist

# ---------------------------------------------------------------
# Python suite: lightweight benchmarks (Chapters 4 and 5)
# ---------------------------------------------------------------
if [ "$SUITE" = "python" ] || [ "$SUITE" = "all" ]; then
  echo
  echo "Python benchmarks"
  run_py baseline_hist \
    chapters/ch04_profiling/python/baseline_hist.py
  run_py bench_transfer \
    chapters/ch05_training/python/bench_transfer.py
  run_py memory_probe \
    chapters/ch05_training/python/memory_probe.py
fi

# ---------------------------------------------------------------
# All: model based benchmarks (Chapters 7 and 9)
# ---------------------------------------------------------------
if [ "$SUITE" = "all" ]; then
  echo
  echo "Model based benchmarks"
  run_py llm_bench chapters/ch07_llm/python/llm_bench.py \
    --quant bf16 --batches 1,4 --out "$OUT/llm_bench.json"

  if [ -f results/ch06/pets.pt ] \
     && [ -f chapters/ch09_tensorrt/models/pets_fp32.onnx ]; then
    run_py compare_backends \
      chapters/ch09_tensorrt/python/compare_backends.py \
      --batches 1,8,32
  else
    skip compare_backends \
      "needs results/ch06/pets.pt and the Chapter 9 ONNX file"
  fi
fi

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
failed=0
{
  echo
  printf '%-28s %s\n' "benchmark" "status"
  printf '%-28s %s\n' "---------" "------"
  for i in "${!NAMES[@]}"; do
    printf '%-28s %s\n' "${NAMES[$i]}" "${STATUS[$i]}"
    if [ "${STATUS[$i]}" = "FAIL" ]; then failed=$((failed + 1)); fi
  done
  echo
  echo "Results saved in $OUT"
} | tee "$OUT/summary.txt"

if [ "$failed" -gt 0 ]; then
  echo "$failed benchmark(s) failed."
  exit 1
fi
exit 0