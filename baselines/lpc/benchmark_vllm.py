import asyncio, aiohttp, json, time, argparse, sys

async def send_request(session, url, prompt, model, req_id):
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 256, 'stream': False}
    start = time.monotonic()
    try:
        async with session.post(f'{url}/v1/chat/completions', json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data = await resp.json()
            ttft = time.monotonic() - start
            if 'error' in data:
                return {'req_id': req_id, 'ttft': ttft, 'error': str(data['error'])}
            return {'req_id': req_id, 'ttft': ttft}
    except Exception as e:
        return {'req_id': req_id, 'ttft': time.monotonic() - start, 'error': str(e)}

def build_sharegpt_prompts(n=500):
    from datasets import load_dataset
    ds = load_dataset('RyokoAI/ShareGPT52K', split='train')
    prompts = []
    for item in ds:
        convs = item.get('conversations', [])
        if not convs or len(convs) < 4:
            continue
        cumul = ''
        for msg in convs:
            role = msg.get('from', 'human')
            val = msg.get('value', '')
            if not val or not isinstance(val, str):
                continue
            cumul += f'\n[{role}]: {val}'
            if role == 'human' and len(cumul) > 200:
                prompts.append(cumul[:3500])
        if len(prompts) >= n:
            break
    return prompts[:n]

async def run_benchmark(url, model, num_requests, output_path, concurrency=1):
    print(f'Building {num_requests} prompts from ShareGPT...')
    prompts = build_sharegpt_prompts(num_requests)
    print(f'Got {len(prompts)} prompts, sending to {url}...')
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, prompt in enumerate(prompts):
            r = await send_request(session, url, prompt, model, i)
            results.append(r)
            if (i + 1) % 50 == 0:
                valid = [x['ttft'] for x in results if 'error' not in x]
                avg = sum(valid) / len(valid) if valid else 0
                print(f'  [{i + 1}/{len(prompts)}] avg_TTFT={avg:.4f}s')
    valid = [x['ttft'] for x in results if 'error' not in x]
    errors = [x for x in results if 'error' in x]
    if valid:
        valid_sorted = sorted(valid)
        stats = {'num_requests': len(results), 'num_success': len(valid), 'num_errors': len(errors), 'avg_ttft': sum(valid) / len(valid), 'p50_ttft': valid_sorted[len(valid) // 2], 'p99_ttft': valid_sorted[min(int(len(valid) * 0.99), len(valid) - 1)], 'min_ttft': valid_sorted[0], 'max_ttft': valid_sorted[-1]}
    else:
        stats = {'error': 'all requests failed'}
    output = {'stats': stats, 'results': results}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 50}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}s')
        else:
            print(f'  {k}: {v}')
    print(f'Saved to {output_path}')
    print(f"{'=' * 50}")
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://localhost:8000')
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--num-requests', type=int, default=500)
    p.add_argument('--output', default='results.json')
    p.add_argument('--concurrency', type=int, default=1)
    args = p.parse_args()
    asyncio.run(run_benchmark(args.url, args.model, args.num_requests, args.output, args.concurrency))
