import json, time, requests, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
random.seed(42)
SHAREGPT_PATH = '/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json'
URL = 'http://localhost:8000/v1/chat/completions'
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
SYSTEM_PROMPTS = ['You are a helpful AI assistant that specializes in coding and software development. You write clean, efficient code and explain complex technical concepts clearly.', 'You are an expert travel planner who creates detailed itineraries for international trips. You consider budget, local customs, weather, and transportation options.', 'You are a professional chef and nutritionist who creates healthy, delicious recipes. You provide detailed ingredient lists, cooking instructions, and nutritional information.', 'You are a financial advisor who helps people manage their money wisely. You explain investment strategies, budgeting techniques, and tax optimization in simple terms.', 'You are a creative writing coach who helps people craft compelling stories. You provide feedback on narrative structure, character development, dialogue, and prose style.']
INTERVALS = [0.02, 0.03, 0.05, 0.08, 0.1]
tag = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
print(f'[1] Loading ShareGPT...')
with open(SHAREGPT_PATH) as f:
    data = json.load(f)
all_requests = []
for item in data:
    convs = item.get('conversations', [])
    if not convs:
        continue
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    messages = [{'role': 'system', 'content': sys_prompt}]
    for msg in convs[:4]:
        role_from = msg.get('from', '')
        content = msg.get('value', '')[:300]
        if role_from in ('human', 'user'):
            messages.append({'role': 'user', 'content': content})
        elif role_from in ('gpt', 'assistant'):
            messages.append({'role': 'assistant', 'content': content})
    if len(messages) < 2:
        continue
    if messages[-1]['role'] != 'user':
        messages.append({'role': 'user', 'content': 'Continue.'})
    all_requests.append(messages)
    if len(all_requests) >= 4928:
        break
print(f'  Prepared {len(all_requests)} requests')

def run_one(reqs, interval):
    ttfts = []
    errors = 0

    def send(messages):
        try:
            t0 = time.time()
            r = requests.post(URL, json={'model': MODEL, 'messages': messages, 'max_tokens': 50, 'stream': True}, timeout=120, stream=True)
            ttft = None
            for line in r.iter_lines():
                if line:
                    d = line.decode('utf-8')
                    if d.startswith('data: ') and d != 'data: [DONE]':
                        if ttft is None:
                            ttft = time.time() - t0
                        break
            for line in r.iter_lines():
                pass
            return ttft if ttft else time.time() - t0
        except:
            return None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = []
        for i, msgs in enumerate(reqs):
            futures.append(ex.submit(send, msgs))
            time.sleep(interval)
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                ttfts.append(r)
            else:
                errors += 1
    elapsed = time.time() - t0
    ttfts.sort()
    n = len(ttfts)
    if n == 0:
        return None
    return {'interval': interval, 'n': n, 'errors': errors, 'mean': sum(ttfts) / n, 'p50': ttfts[n // 2], 'p90': ttfts[int(n * 0.9)], 'p99': ttfts[int(n * 0.99)], 'elapsed': elapsed, 'ttfts': ttfts}
print(f'\n[2] Sweeping intervals for evictor={tag}')
print(f'    Intervals: {INTERVALS}')
print(f"{'=' * 80}")
all_results = []
for interval in INTERVALS:
    print(f'\n  --- interval={interval}s ---')
    result = run_one(all_requests, interval)
    if result is None:
        print(f'    FAILED')
        continue
    result['tag'] = tag
    all_results.append(result)
    print(f"    Mean={result['mean'] * 1000:.1f}ms  P50={result['p50'] * 1000:.1f}ms  P90={result['p90'] * 1000:.1f}ms  P99={result['p99'] * 1000:.1f}ms  N={result['n']}  Duration={result['elapsed']:.0f}s")
    outpath = f"/workspace/ttft_sweep_{tag}_i{str(interval).replace('.', '')}.json"
    with open(outpath, 'w') as f:
        json.dump(result, f)
    print(f'    Cooling down 5s...')
    time.sleep(5)
print(f"\n{'=' * 80}")
print(f'  SWEEP SUMMARY (evictor={tag})')
print(f"{'=' * 80}")
print(f"  {'Interval':>10} {'Mean':>10} {'P50':>10} {'P90':>10} {'P99':>10} {'N':>6}")
print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 6}")
for r in all_results:
    print(f"  {r['interval']:>8.2f}s {r['mean'] * 1000:>8.1f}ms {r['p50'] * 1000:>8.1f}ms {r['p90'] * 1000:>8.1f}ms {r['p99'] * 1000:>8.1f}ms {r['n']:>6}")
print(f"{'=' * 80}")
summary_path = f'/workspace/ttft_sweep_{tag}_summary.json'
summary = [{'interval': r['interval'], 'mean': r['mean'], 'p50': r['p50'], 'p90': r['p90'], 'p99': r['p99'], 'n': r['n']} for r in all_results]
with open(summary_path, 'w') as f:
    json.dump(summary, f)
print(f'  Saved summary to {summary_path}')
