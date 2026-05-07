import argparse
import json
import os
import random
import signal
import string
import subprocess
import sys
import time
import requests
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
DATASET = '/workspace/LPC/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json'
MLP_PATH = '/workspace/session_predictor_mlp.pt'
PORT = 8000
URL = f'http://localhost:{PORT}/v1/chat/completions'
HEALTH_URL = f'http://localhost:{PORT}/health'
DEFAULT_INTERVALS = [0.03, 0.05, 0.1, 0.5]
N_REQUESTS = 200
GPU_BLOCKS = 200
WARMUP = 20
SYSTEM_PROMPTS = ['You are a helpful coding assistant specializing in Python.', 'You are a creative writing tutor who helps students improve their essays.', 'You are a financial advisor providing investment guidance.', 'You are a travel guide recommending destinations worldwide.', 'You are a health and nutrition expert.']
PREDICTOR_HOOK_CODE = "import torch\nimport torch.nn as nn\nimport threading\n\nclass SessionPredictorMLP(nn.Module):\n\n    def __init__(self, input_dim=4096, hidden_dim=256):\n        super().__init__()\n        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))\n\n    def forward(self, x):\n        return self.net(x)\n_predictor_model = None\n_predictor_lock = threading.Lock()\n_predictor_enabled = {enabled}\n\ndef get_predictor():\n    global _predictor_model\n    if _predictor_model is None and _predictor_enabled:\n        with _predictor_lock:\n            if _predictor_model is None:\n                try:\n                    _predictor_model = SessionPredictorMLP(4096, 256)\n                    state = torch.load('{mlp_path}', map_location='cpu', weights_only=True)\n                    _predictor_model.load_state_dict(state)\n                    _predictor_model.eval()\n                    print('[SAECache] Session predictor loaded successfully')\n                except Exception as e:\n                    print(f'[SAECache] Failed to load predictor: {{e}}')\n                    _predictor_enabled = False\n    return _predictor_model\n\ndef predict_session_type(hidden_state_tensor):\n    model = get_predictor()\n    if model is None:\n        return None\n    try:\n        with torch.no_grad():\n            if hidden_state_tensor.dim() == 2:\n                h = hidden_state_tensor[-1:, :]\n            else:\n                h = hidden_state_tensor.unsqueeze(0)\n            logits = model(h.float().cpu())\n            pred = torch.argmax(logits, dim=1).item()\n            return pred\n    except Exception:\n        return None"

def load_dataset():
    with open(DATASET, 'r') as f:
        data = json.load(f)
    conversations = []
    for conv in data:
        turns = conv.get('conversations', [])
        if len(turns) >= 2 and turns[0].get('from') == 'human':
            conversations.append(turns)
    random.seed(42)
    random.shuffle(conversations)
    return conversations[:2000]

def build_requests(conversations, n_requests):
    reqs = []
    random.seed(42)
    for i in range(n_requests):
        sys_prompt = random.choice(SYSTEM_PROMPTS)
        conv = conversations[i % len(conversations)]
        if i % 2 == 0 and len(conv) >= 4:
            messages = [{'role': 'system', 'content': sys_prompt}]
            for j, turn in enumerate(conv[:4]):
                role = 'user' if turn['from'] == 'human' else 'assistant'
                messages.append({'role': role, 'content': turn['value'][:500]})
            if messages[-1]['role'] == 'assistant':
                messages.append({'role': 'user', 'content': 'Continue.'})
        else:
            messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': conv[0]['value'][:500]}]
        reqs.append(messages)
    return reqs

def measure_ttft(messages):
    try:
        t0 = time.time()
        r = requests.post(URL, json={'model': MODEL, 'messages': messages, 'max_tokens': 1, 'stream': True}, timeout=120, stream=True)
        for line in r.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                    ttft = time.time() - t0
                    for _ in r.iter_lines():
                        pass
                    return ttft
        return time.time() - t0
    except Exception as e:
        print(f'  Error: {e}')
        return None

def start_server(predictor_enabled):
    env = os.environ.copy()
    env['VLLM_USE_V1'] = '0'
    if predictor_enabled:
        env['SAGECACHE_PREDICTOR'] = '1'
    else:
        env['SAGECACHE_PREDICTOR'] = '0'
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL, '--enable-prefix-caching', '--gpu-memory-utilization', '0.9', '--num-gpu-blocks-override', str(GPU_BLOCKS), '--max-model-len', '4096', '--port', str(PORT)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"  Waiting for server (predictor={('ON' if predictor_enabled else 'OFF')})...")
    for attempt in range(120):
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                print(f'  Server ready after {attempt + 1}s')
                return proc
        except:
            pass
        time.sleep(1)
    print('  ERROR: Server failed to start!')
    proc.kill()
    return None

def stop_server(proc):
    if proc:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(3)

def run_benchmark(requests_list, interval, label):
    print(f'\n  [{label}] interval={interval}s, {len(requests_list)} requests')
    print(f'    Warming up ({WARMUP} requests)...')
    for i in range(min(WARMUP, len(requests_list))):
        measure_ttft(requests_list[i])
        time.sleep(0.05)
    ttfts = []
    for i, msgs in enumerate(requests_list):
        ttft = measure_ttft(msgs)
        if ttft is not None:
            ttfts.append(ttft * 1000)
        time.sleep(interval)
        if (i + 1) % 50 == 0:
            print(f'    Progress: {i + 1}/{len(requests_list)}')
    if not ttfts:
        return None
    ttfts.sort()
    n = len(ttfts)
    result = {'mean': sum(ttfts) / n, 'p50': ttfts[int(n * 0.5)], 'p90': ttfts[int(n * 0.9)], 'p99': ttfts[int(n * 0.99)] if n >= 100 else ttfts[-1], 'n': n}
    print(f"    Mean={result['mean']:.1f}ms  P50={result['p50']:.1f}ms  P90={result['p90']:.1f}ms  P99={result['p99']:.1f}ms")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intervals', type=str, default='0.03,0.05,0.1,0.5', help='Comma-separated request intervals in seconds')
    parser.add_argument('--n-requests', type=int, default=200)
    args = parser.parse_args()
    intervals = [float(x) for x in args.intervals.split(',')]
    n_requests = args.n_requests
    print('=' * 70)
    print('  SAECache Predictor Ablation')
    print(f'  Intervals: {intervals}')
    print(f'  Requests per interval: {n_requests}')
    print('=' * 70)
    print('\n  Loading dataset...')
    conversations = load_dataset()
    requests_list = build_requests(conversations, n_requests)
    print(f'  Built {len(requests_list)} requests')
    results = {}
    print('\n' + '=' * 70)
    print('  Phase 1: SAECache (no predictor)')
    print('=' * 70)
    proc = start_server(predictor_enabled=False)
    if proc is None:
        print('Server failed to start, aborting.')
        return
    results['no_predictor'] = {}
    for interval in intervals:
        result = run_benchmark(requests_list, interval, 'no_predictor')
        if result:
            results['no_predictor'][str(interval)] = result
    stop_server(proc)
    print('\n' + '=' * 70)
    print('  Phase 2: SAECache + Predictor')
    print('=' * 70)
    proc = start_server(predictor_enabled=True)
    if proc is None:
        print('Server failed to start, aborting.')
        return
    results['with_predictor'] = {}
    for interval in intervals:
        result = run_benchmark(requests_list, interval, 'with_predictor')
        if result:
            results['with_predictor'][str(interval)] = result
    stop_server(proc)
    print('\n' + '=' * 70)
    print('  COMPARISON: SAECache vs SAECache + Predictor')
    print('=' * 70)
    print(f"  {'Interval':>10} | {'Metric':>6} | {'No Pred':>10} | {'+ Pred':>10} | {'Diff':>10}")
    print(f"  {'-' * 10}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    for interval in intervals:
        si = str(interval)
        if si in results['no_predictor'] and si in results['with_predictor']:
            no_p = results['no_predictor'][si]
            wi_p = results['with_predictor'][si]
            for metric in ['mean', 'p50', 'p90', 'p99']:
                v_no = no_p[metric]
                v_wi = wi_p[metric]
                diff_pct = (v_wi - v_no) / v_no * 100
                sign = '+' if diff_pct > 0 else ''
                print(f'  {interval:>10.3f} | {metric:>6} | {v_no:>8.1f}ms | {v_wi:>8.1f}ms | {sign}{diff_pct:>7.1f}%')
            print(f"  {'-' * 10}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    outpath = '/workspace/ttft_predictor_ablation.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {outpath}')
if __name__ == '__main__':
    main()
