#!/bin/bash
# run_experiment.sh — LRU vs SAECache with forced cache pressure
# 用法: cd /tmp && bash /workspace/LPC/run_experiment.sh

set -e
EVICTOR="/usr/local/lib/python3.11/dist-packages/vllm/core/evictor.py"
BENCH="/workspace/LPC/benchmark_vllm.py"
OUTDIR="/workspace/LPC"

# Key parameter: force very small KV cache (50 blocks × 16 tokens = 800 tokens of cache)
# This guarantees eviction will happen on almost every request
NUM_GPU_BLOCKS=256

echo "=============================================="
echo "  SAECache vLLM Experiment"
echo "  GPU blocks: $NUM_GPU_BLOCKS (forced small)"
echo "=============================================="

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

########################################
# Phase 1: LRU Baseline
########################################
echo ""
echo "[Phase 1/2] Setting evictor to LRU..."

python3 -c "
path = '$EVICTOR'
with open(path) as f: c = f.read()
c = c.replace('DISABLED', 'lru')
c = c.replace(\"elif True:  # Force SAECache\", \"elif eviction_algorithm == 'saecache':\")
with open(path, 'w') as f: f.write(c)
print('  ✓ evictor = LRU')
"

echo "[Phase 1/2] Starting LRU server..."
cd /tmp
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --num-gpu-blocks-override $NUM_GPU_BLOCKS \
    --port 8000 > $OUTDIR/vllm_exp_lru.log 2>&1 &

echo "  Waiting for server (max 180s)..."
for i in $(seq 1 36); do
    sleep 5
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
        echo "  ✓ Server ready after $((i*5))s"
        break
    fi
    if [ $i -eq 36 ]; then
        echo "  ✗ Server failed. Log:"
        tail -10 $OUTDIR/vllm_exp_lru.log
        exit 1
    fi
done

echo "[Phase 1/2] Running LRU benchmark (1000 reqs, concurrency=4)..."
python3 $BENCH \
    --output $OUTDIR/results_exp_lru.json \
    --num-requests 1000 \
    --concurrency 4

# Check if eviction happened
echo "  Eviction count in LRU log:"
grep -c -i "evict" $OUTDIR/vllm_exp_lru.log || echo "  0"

echo "[Phase 1/2] Stopping LRU server..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

########################################
# Phase 2: SAECache
########################################
echo ""
echo "[Phase 2/2] Setting evictor to SAECache..."

python3 -c "
path = '$EVICTOR'
with open(path) as f: c = f.read()
c = c.replace(\"if eviction_algorithm == 'lru':\", \"if eviction_algorithm == 'DISABLED':\")
c = c.replace(\"elif eviction_algorithm == 'saecache':\", \"elif True:  # Force SAECache\")
with open(path, 'w') as f: f.write(c)
print('  ✓ evictor = SAECache')
"

echo "[Phase 2/2] Starting SAECache server..."
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --num-gpu-blocks-override $NUM_GPU_BLOCKS \
    --port 8000 > $OUTDIR/vllm_exp_learned.log 2>&1 &

echo "  Waiting for server (max 180s)..."
for i in $(seq 1 36); do
    sleep 5
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
        echo "  ✓ Server ready after $((i*5))s"
        break
    fi
    if [ $i -eq 36 ]; then
        echo "  ✗ Server failed. Log:"
        tail -10 $OUTDIR/vllm_exp_learned.log
        exit 1
    fi
done

echo "[Phase 2/2] Running SAECache benchmark (1000 reqs, concurrency=4)..."
python3 $BENCH \
    --output $OUTDIR/results_exp_learned.json \
    --num-requests 1000 \
    --concurrency 4

# Check if eviction happened
echo "  Eviction count in SAECache log:"
grep -c -i "evict" $OUTDIR/vllm_exp_learned.log || echo "  0"

echo "[Phase 2/2] Stopping SAECache server..."
pkill -f "vllm.entrypoints" 2>/dev/null || true

########################################
# Results
########################################
echo ""
echo "=============================================="
echo "  RESULTS (gpu_blocks=$NUM_GPU_BLOCKS)"
echo "=============================================="
python3 -c "
import json
for name, label in [('results_exp_lru.json','LRU'), ('results_exp_learned.json','SAECache')]:
    try:
        with open(f'$OUTDIR/{name}') as f:
            s = json.load(f)['stats']
        print(f'\n{label}:')
        for k in ['avg_ttft','p50_ttft','p99_ttft','min_ttft','max_ttft','num_success','num_errors']:
            v = s.get(k, 'N/A')
            print(f'  {k}: {v:.4f}s' if isinstance(v,float) else f'  {k}: {v}')
    except Exception as e:
        print(f'{label}: ERROR - {e}')

try:
    with open(f'$OUTDIR/results_exp_lru.json') as f: lru = json.load(f)['stats']
    with open(f'$OUTDIR/results_exp_learned.json') as f: lc = json.load(f)['stats']
    if lc.get('avg_ttft',0) > 0:
        print(f'\nSpeedup (LRU / SAECache):')
        print(f'  avg_TTFT: {lru[\"avg_ttft\"]/lc[\"avg_ttft\"]:.2f}x')
        print(f'  p50_TTFT: {lru[\"p50_ttft\"]/lc[\"p50_ttft\"]:.2f}x')
        print(f'  p99_TTFT: {lru[\"p99_ttft\"]/lc[\"p99_ttft\"]:.2f}x')
except: pass
"
echo ""
echo "=============================================="

# Restore evictor
python3 -c "
path = '$EVICTOR'
with open(path) as f: c = f.read()
c = c.replace('DISABLED', 'lru')
c = c.replace(\"elif True:  # Force SAECache\", \"elif eviction_algorithm == 'saecache':\")
with open(path, 'w') as f: f.write(c)
print('✓ evictor restored')
"