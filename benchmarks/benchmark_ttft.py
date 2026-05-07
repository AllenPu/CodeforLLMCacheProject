import json, time, requests, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
with open('/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json') as f:
    data = json.load(f)
url = 'http://localhost:8000/v1/chat/completions'
model = '/workspace/models/Qwen2.5-1.5B-Instruct'
all_requests = []
for item in data[:5000]:
    convs = item.get('conversations', [])
    if not convs or len(convs) < 2:
        continue
    messages = [{'role': 'system', 'content': 'You are a helpful AI assistant.'}]
    for msg in convs[:4]:
        role = 'user' if msg.get('from') in ('human', 'user') else 'assistant'
        content = msg.get('value', '')[:300]
        if content:
            messages.append({'role': role, 'content': content})
    if messages[-1]['role'] != 'user':
        messages.append({'role': 'user', 'content': 'Continue.'})
    all_requests.append(messages)
print(f'Prepared {len(all_requests)} requests')
ttfts = []
errors = 0

def send_request(messages):
    try:
        t0 = time.time()
        r = requests.post(url, json={'model': model, 'messages': messages, 'max_tokens': 50, 'stream': True}, timeout=120, stream=True)
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
    except Exception as e:
        return None
start = time.time()
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = []
    for i, msgs in enumerate(all_requests):
        futures.append(executor.submit(send_request, msgs))
        time.sleep(0.03)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f'Submitted {i + 1}/{len(all_requests)}, {elapsed:.0f}s')
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
    print(f"\n{'=' * 50}")
    print(f'  RESULTS')
    print(f"{'=' * 50}")
    print(f'  Requests:  {n} success, {errors} errors')
    print(f'  Duration:  {elapsed:.0f}s')
    print(f'  Mean TTFT: {mean_ttft * 1000:.1f} ms')
    print(f'  P50 TTFT:  {p50 * 1000:.1f} ms')
    print(f'  P90 TTFT:  {p90 * 1000:.1f} ms')
    print(f'  P99 TTFT:  {p99 * 1000:.1f} ms')
    print(f"{'=' * 50}")
    tag = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
    with open(f'/workspace/ttft_{tag}.json', 'w') as f:
        json.dump({'tag': tag, 'ttfts': ttfts, 'mean': mean_ttft, 'p50': p50, 'p90': p90, 'p99': p99, 'n': n, 'errors': errors}, f)
    print(f'  Saved to /workspace/ttft_{tag}.json')
