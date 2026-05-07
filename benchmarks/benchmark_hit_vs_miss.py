import json, time, requests, sys, random, string
URL = 'http://localhost:8000/v1/chat/completions'
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
N_REPEATS = 20
random.seed(42)

def generate_text(n_words):
    words = []
    for _ in range(n_words):
        word_len = random.randint(3, 8)
        words.append(''.join(random.choices(string.ascii_lowercase, k=word_len)))
    return ' '.join(words)

def measure_ttft(messages):
    try:
        t0 = time.time()
        r = requests.post(URL, json={'model': MODEL, 'messages': messages, 'max_tokens': 10, 'stream': True}, timeout=120, stream=True)
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
        print(f'  Error: {e}')
        return None
CONTEXT_LENGTHS = [1000, 3000]
PROMPT_LENGTHS = [50, 500, 1000]
print('=' * 70)
print('  Cache Hit vs Miss TTFT (reproducing LPC Figure 7)')
print('=' * 70)
results = {}
for ctx_len in CONTEXT_LENGTHS:
    for prompt_len in PROMPT_LENGTHS:
        key = f'ctx{ctx_len}_prompt{prompt_len}'
        print(f'\n  Config: context={ctx_len} tokens, new_prompt={prompt_len} tokens')
        ctx_words = int(ctx_len / 1.3)
        prompt_words = int(prompt_len / 1.3)
        miss_ttfts = []
        hit_ttfts = []
        for rep in range(N_REPEATS):
            unique_id = f'{key}_rep{rep}_{random.randint(0, 999999)}'
            context_text = f'Session {unique_id}. ' + generate_text(ctx_words)
            prompt_text = generate_text(prompt_words)
            messages_base = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': context_text[:int(ctx_len * 3)]}, {'role': 'assistant', 'content': generate_text(int(ctx_words * 0.3))}]
            messages_with_prompt = messages_base + [{'role': 'user', 'content': prompt_text[:int(prompt_len * 3)]}]
            ttft_miss = measure_ttft(messages_with_prompt)
            if ttft_miss:
                miss_ttfts.append(ttft_miss)
            time.sleep(0.1)
            ttft_hit = measure_ttft(messages_with_prompt)
            if ttft_hit:
                hit_ttfts.append(ttft_hit)
            time.sleep(0.1)
        if miss_ttfts and hit_ttfts:
            mean_miss = sum(miss_ttfts) / len(miss_ttfts)
            mean_hit = sum(hit_ttfts) / len(hit_ttfts)
            reduction = (1 - mean_hit / mean_miss) * 100
            results[key] = {'context_len': ctx_len, 'prompt_len': prompt_len, 'mean_miss': mean_miss, 'mean_hit': mean_hit, 'reduction': reduction, 'n_miss': len(miss_ttfts), 'n_hit': len(hit_ttfts)}
            print(f'    Miss: {mean_miss * 1000:.1f} ms ({len(miss_ttfts)} samples)')
            print(f'    Hit:  {mean_hit * 1000:.1f} ms ({len(hit_ttfts)} samples)')
            print(f'    Reduction: {reduction:.1f}%')
print(f"\n{'=' * 70}")
print(f'  SUMMARY (TTFT in ms)')
print(f"{'=' * 70}")
print(f"  {'Context':>10} {'Prompt':>10} {'Miss':>10} {'Hit':>10} {'Reduction':>10}")
print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
for key in sorted(results.keys()):
    r = results[key]
    print(f"  {r['context_len']:>8}tk {r['prompt_len']:>8}tk {r['mean_miss'] * 1000:>8.1f}ms {r['mean_hit'] * 1000:>8.1f}ms {r['reduction']:>8.1f}%")
print(f"{'=' * 70}")
outpath = '/workspace/ttft_hit_vs_miss.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f'  Saved to {outpath}')
