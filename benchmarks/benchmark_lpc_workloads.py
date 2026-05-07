import json, time, random, sys
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed
URL = 'http://localhost:8000/v1/chat/completions'
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
DATASET_PATH = '/workspace/LPC/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json'
ALT_PATH = '/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json'
SYSTEM_PROMPTS = ['You are a helpful AI assistant that specializes in coding and software development. You write clean, efficient code and explain complex technical concepts clearly.', 'You are an expert travel planner who creates detailed itineraries for international trips. You consider budget, local customs, weather, and transportation options.', 'You are a professional chef and nutritionist who creates healthy, delicious recipes. You provide detailed ingredient lists, cooking instructions, and nutritional information.', 'You are a financial advisor who helps people manage their money wisely. You explain investment strategies, budgeting techniques, and tax optimization in simple terms.', 'You are a creative writing coach who helps people craft compelling stories. You provide feedback on narrative structure, character development, dialogue, and prose style.']
random.seed(42)
import os
ds_path = DATASET_PATH if os.path.exists(DATASET_PATH) else ALT_PATH
with open(ds_path) as f:
    raw_data = json.load(f)
single_turn_convs = []
multi_turn_convs = []
for item in raw_data:
    convs = item.get('conversations', [])
    if not convs:
        continue
    user_turns = sum((1 for m in convs if m.get('from', m.get('role', '')) in ('human', 'user')))
    if user_turns >= 2:
        multi_turn_convs.append(convs)
    elif user_turns == 1:
        single_turn_convs.append(convs)
random.shuffle(single_turn_convs)
random.shuffle(multi_turn_convs)
print(f'Available: {len(single_turn_convs)} single-turn, {len(multi_turn_convs)} multi-turn')

def build_single_turn_request(conv):
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    user_msg = ''
    for msg in conv:
        if msg.get('from', msg.get('role', '')) in ('human', 'user'):
            user_msg = msg.get('value', msg.get('content', ''))[:300]
            break
    if not user_msg:
        user_msg = 'Hello.'
    return [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_msg}]

def build_multi_turn_request(conv):
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    messages = [{'role': 'system', 'content': sys_prompt}]
    for msg in conv[:6]:
        from_field = msg.get('from', msg.get('role', ''))
        content = msg.get('value', msg.get('content', ''))[:300]
        if from_field in ('human', 'user'):
            messages.append({'role': 'user', 'content': content})
        elif from_field in ('gpt', 'assistant'):
            messages.append({'role': 'assistant', 'content': content})
    if messages[-1]['role'] != 'user':
        messages.append({'role': 'user', 'content': 'Continue.'})
    return messages

def build_workload(name, n_total, multi_ratio):
    n_multi = int(n_total * multi_ratio)
    n_single = n_total - n_multi
    reqs = []
    for i in range(n_multi):
        conv = multi_turn_convs[i % len(multi_turn_convs)]
        reqs.append(build_multi_turn_request(conv))
    for i in range(n_single):
        conv = single_turn_convs[i % len(single_turn_convs)]
        reqs.append(build_single_turn_request(conv))
    random.shuffle(reqs)
    return reqs
WORKLOADS = {'tool_use': {'n': 282, 'multi_ratio': 0.0}, 'multi_turn_dominant': {'n': 1000, 'multi_ratio': 0.8}, 'balanced': {'n': 1000, 'multi_ratio': 0.5}, 'single_turn_dominant': {'n': 1000, 'multi_ratio': 0.2}}

def measure_ttft(messages):
    try:
        t0 = time.time()
        r = http_requests.post(URL, json={'model': MODEL, 'messages': messages, 'max_tokens': 50, 'stream': True}, timeout=120, stream=True)
        for line in r.iter_lines():
            if line:
                d = line.decode('utf-8')
                if d.startswith('data: ') and d != 'data: [DONE]':
                    ttft = time.time() - t0
                    for _ in r.iter_lines():
                        pass
                    return ttft
        return time.time() - t0
    except:
        return None

def run_workload(name, requests_list, interval=0.03):
    print(f'\n  --- {name} ({len(requests_list)} reqs, interval={interval}s) ---')
    print(f'    Warming up (20 reqs)...')
    for i in range(min(20, len(requests_list))):
        measure_ttft(requests_list[i])
        time.sleep(0.05)
    ttfts = []
    errors = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = []
        for i, msgs in enumerate(requests_list):
            futures.append(ex.submit(measure_ttft, msgs))
            time.sleep(interval)
            if (i + 1) % 200 == 0:
                print(f'    {i + 1}/{len(requests_list)}')
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                ttfts.append(r)
            else:
                errors += 1
    if not ttfts:
        print(f'    ALL FAILED')
        return None
    ttfts.sort()
    n = len(ttfts)
    stats = {'workload': name, 'n': n, 'errors': errors, 'mean': sum(ttfts) / n, 'p50': ttfts[n // 2], 'p90': ttfts[int(n * 0.9)], 'p99': ttfts[int(n * 0.99)] if n >= 100 else ttfts[-1]}
    print(f"    Mean={stats['mean'] * 1000:.1f}ms  P50={stats['p50'] * 1000:.1f}ms  P90={stats['p90'] * 1000:.1f}ms  N={n}  Errors={errors}")
    return stats

def main():
    try:
        r = http_requests.get('http://localhost:8000/health', timeout=5)
        if r.status_code != 200:
            print('Server not healthy!')
            sys.exit(1)
    except:
        print('Server not running! Start it first.')
        sys.exit(1)
    print('=' * 60)
    print('  LPC Workload Composition Benchmark')
    print('=' * 60)
    results = {}
    for wl_name, wl_config in WORKLOADS.items():
        reqs = build_workload(wl_name, wl_config['n'], wl_config['multi_ratio'])
        stats = run_workload(wl_name, reqs, interval=0.03)
        if stats:
            results[wl_name] = stats
    print(f"\n{'=' * 60}")
    print(f'  SUMMARY (LPC)')
    print(f"{'=' * 60}")
    print(f"  {'Workload':<25} {'Mean TTFT':>12} {'P50':>12} {'P90':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")
    for name, r in results.items():
        print(f"  {name:<25} {r['mean'] * 1000:>10.1f}ms {r['p50'] * 1000:>10.1f}ms {r['p90'] * 1000:>10.1f}ms")
    outpath = '/workspace/results/lpc_workload_compositions.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved to {outpath}')
if __name__ == '__main__':
    main()
