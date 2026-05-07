import json, time, requests, sys, random, os
from concurrent.futures import ThreadPoolExecutor, as_completed
random.seed(42)
URL = 'http://localhost:8000/v1/chat/completions'
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
MAX_REQUESTS = 3000
SYSTEM_PROMPTS = ['You are a helpful AI assistant that specializes in coding and software development. You write clean, efficient code and explain complex technical concepts clearly.', 'You are an expert travel planner who creates detailed itineraries for international trips. You consider budget, local customs, weather, and transportation options.', 'You are a professional chef and nutritionist who creates healthy, delicious recipes. You provide detailed ingredient lists, cooking instructions, and nutritional information.', 'You are a financial advisor who helps people manage their money wisely. You explain investment strategies, budgeting techniques, and tax optimization in simple terms.', 'You are a creative writing coach who helps people craft compelling stories. You provide feedback on narrative structure, character development, dialogue, and prose style.']
DATASETS = {'chatbot_arena': '/workspace/datasets/converted/chatbot_arena.json', 'lmsys': '/workspace/datasets/converted/lmsys_chat.json', 'sharegpt': '/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json'}
tag = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
interval = 0.03
for i, arg in enumerate(sys.argv):
    if arg == '--interval' and i + 1 < len(sys.argv):
        interval = float(sys.argv[i + 1])
print(f"{'=' * 70}")
print(f'  Unified 3-Dataset Benchmark')
print(f'  Evictor: {tag}')
print(f'  Request interval: {interval}s')
print(f"{'=' * 70}")

def load_dataset(name, path, max_n):
    with open(path) as f:
        data = json.load(f)
    all_requests = []
    for item in data[:max_n * 2]:
        convs = item.get('conversations', [])
        if not convs:
            continue
        sys_prompt = random.choice(SYSTEM_PROMPTS)
        messages = [{'role': 'system', 'content': sys_prompt}]
        for msg in convs[:4]:
            from_field = msg.get('from', msg.get('role', ''))
            content = msg.get('value', msg.get('content', ''))[:300]
            if from_field in ('human', 'user'):
                role = 'user'
            elif from_field in ('gpt', 'assistant'):
                role = 'assistant'
            else:
                continue
            if content:
                messages.append({'role': role, 'content': content})
        if len(messages) < 2:
            continue
        if messages[-1]['role'] != 'user':
            messages.append({'role': 'user', 'content': 'Continue.'})
        all_requests.append(messages)
        if len(all_requests) >= max_n:
            break
    return all_requests

def run_benchmark(all_requests, dataset_name, tag, interval):
    ttfts = []
    errors = 0

    def send_request(messages):
        try:
            t0 = time.time()
            r = requests.post(URL, json={'model': MODEL, 'messages': messages, 'max_tokens': 50, 'stream': True}, timeout=120, stream=True)
            ttft = None
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                        if ttft is None:
                            ttft = time.time() - t0
                        break
            for line in r.iter_lines():
                pass
            return ttft if ttft else time.time() - t0
        except:
            return None
    start = time.time()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i, msgs in enumerate(all_requests):
            futures.append(executor.submit(send_request, msgs))
            time.sleep(interval)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print(f'    Submitted {i + 1}/{len(all_requests)}, {elapsed:.0f}s')
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                ttfts.append(result)
            else:
                errors += 1
    elapsed = time.time() - start
    ttfts.sort()
    n = len(ttfts)
    if n > 0:
        mean_ttft = sum(ttfts) / n
        p50 = ttfts[n // 2]
        p90 = ttfts[int(n * 0.9)]
        p99 = ttfts[int(n * 0.99)]
    else:
        mean_ttft = p50 = p90 = p99 = 0
    result = {'dataset': dataset_name, 'tag': tag, 'interval': interval, 'n': n, 'errors': errors, 'mean': mean_ttft, 'p50': p50, 'p90': p90, 'p99': p99, 'elapsed': elapsed, 'ttfts': ttfts}
    outpath = f'/workspace/ttft_3ds_{dataset_name}_{tag}.json'
    with open(outpath, 'w') as f:
        json.dump(result, f)
    return result
all_results = []
for ds_name, ds_path in DATASETS.items():
    if not os.path.exists(ds_path):
        print(f'\n  SKIP {ds_name}: file not found')
        continue
    print(f'\n  Loading {ds_name}...')
    reqs = load_dataset(ds_name, ds_path, MAX_REQUESTS)
    print(f'  Prepared {len(reqs)} requests')
    print(f'  Running benchmark...')
    result = run_benchmark(reqs, ds_name, tag, interval)
    all_results.append(result)
    print(f"    Mean TTFT: {result['mean'] * 1000:.1f} ms")
    print(f"    P50 TTFT:  {result['p50'] * 1000:.1f} ms")
    print(f"    P90 TTFT:  {result['p90'] * 1000:.1f} ms")
    print(f"    P99 TTFT:  {result['p99'] * 1000:.1f} ms")
    print(f"    Requests:  {result['n']} ok, {result['errors']} errors")
print(f"\n{'=' * 70}")
print(f'  SUMMARY (evictor={tag}, interval={interval}s)')
print(f"{'=' * 70}")
print(f"  {'Dataset':<20} {'Mean TTFT':>12} {'P50':>10} {'P90':>10} {'P99':>10} {'N':>6}")
print(f"  {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 6}")
for r in all_results:
    print(f"  {r['dataset']:<20} {r['mean'] * 1000:>10.1f}ms {r['p50'] * 1000:>8.1f}ms {r['p90'] * 1000:>8.1f}ms {r['p99'] * 1000:>8.1f}ms {r['n']:>6}")
print(f"{'=' * 70}")
print(f'  Results saved to /workspace/ttft_3ds_*_{tag}.json')
