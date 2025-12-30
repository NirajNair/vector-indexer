#!/bin/bash
# Run the official Faiss-style benchmark for vector_indexer
# Uses the official eval_setting() methodology with min_test_duration averaging

set -e

echo "=============================================="
echo "Vector Indexer - Official Faiss Benchmark"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check io_uring availability
echo ""
echo "Checking io_uring availability..."

if [ -f /proc/sys/kernel/io_uring_disabled ]; then
    IO_URING_DISABLED=$(cat /proc/sys/kernel/io_uring_disabled)
    if [ "$IO_URING_DISABLED" = "2" ]; then
        echo -e "${RED}ERROR: io_uring is disabled on this system (io_uring_disabled=2)${NC}"
        echo "This benchmark requires io_uring to function."
        echo ""
        echo "Possible solutions:"
        echo "  1. Update Docker Desktop to latest version"
        echo "  2. Run container with --privileged flag"
        echo "  3. Use a Linux VM with kernel 5.11+"
        exit 1
    elif [ "$IO_URING_DISABLED" = "1" ]; then
        echo -e "${YELLOW}WARNING: io_uring is partially disabled (unprivileged)${NC}"
        echo "Running with reduced capabilities..."
    fi
fi

# Check kernel version
KERNEL_VERSION=$(uname -r | cut -d'.' -f1-2)
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d'.' -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d'.' -f2)

if [ "$KERNEL_MAJOR" -lt 5 ] || ([ "$KERNEL_MAJOR" -eq 5 ] && [ "$KERNEL_MINOR" -lt 11 ]); then
    echo -e "${YELLOW}WARNING: Kernel version $KERNEL_VERSION may have limited io_uring support${NC}"
    echo "Recommended: kernel 5.11 or later"
fi

echo -e "${GREEN}io_uring check passed${NC}"

# Parse arguments with defaults
N=${N:-100000}
D=${D:-128}
NQ=${NQ:-1000}
K=${K:-100}  # Default 100 for R@1, R@10, R@100
NPROBES=${NPROBES:-"1,2,4,8,16,32,64"}
MIN_TEST_DURATION=${MIN_TEST_DURATION:-3.0}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/faiss_bench_results"}
WORK_DIR=${WORK_DIR:-"/tmp/vector_indexer_bench"}
BACKEND=${BACKEND:-"both"}

echo ""
echo "Benchmark configuration (Official Faiss Methodology):"
echo "  Backend: $BACKEND"
echo "  N: $N"
echo "  D: $D"
echo "  NQ: $NQ"
echo "  K: $K (for R@1, R@10, R@100)"
echo "  NPROBES: $NPROBES"
echo "  MIN_TEST_DURATION: ${MIN_TEST_DURATION}s"
echo "  SEED: $SEED"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  WORK_DIR: $WORK_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORK_DIR"

# Add the benchmark directory to Python path
export PYTHONPATH="/workspace/bench/faiss_bench_official:$PYTHONPATH"

# Run the benchmark
echo "Starting official Faiss-style benchmark..."
echo ""

python3 /workspace/bench/faiss_bench_official/bench_all_ivf.py \
    --backend "$BACKEND" \
    --n "$N" \
    --d "$D" \
    --nq "$NQ" \
    --k "$K" \
    --nprobes "$NPROBES" \
    --min_test_duration "$MIN_TEST_DURATION" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    --work-dir "$WORK_DIR"

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Show summary
if [ -f "$OUTPUT_DIR/faiss_bench_results.md" ]; then
    echo ""
    echo "Summary:"
    cat "$OUTPUT_DIR/faiss_bench_results.md"
fi

