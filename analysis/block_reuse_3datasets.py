import json, time, requests, sys, random
from collections import defaultdict
random.seed(42)
URL = 'http://localhost:8000/v1/chat/completions'
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
SYSTEM_PROMPTS = ['You are a helpful AI assistant that specializes in coding and software development.', 'You are an expert travel planner who creates detailed itineraries.', 'You are a professional chef and nutritionist who creates healthy recipes.', 'You are a financial advisor who helps people manage their money wisely.', 'You are a creative writing coach who helps people craft compelling stories.']
DATASETS = {'ShareGPT': '/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json', 'LMSys': '/workspace/datasets/converted/lmsys_chat.json', 'Chatbot-Arena': '/workspace/datasets/converted/chatbot_arena.json'}
N_REQUESTS = 2000

def analyze_reuse(data, dataset_name, n_requests):
    token_stats = {'system_prompt': {'blocks': 0, 'hits': 0}, 'user_query': {'blocks': 0, 'hits': 0}, 'tool_output': {'blocks': 0, 'hits': 0}, 'response': {'blocks': 0, 'hits': 0}, 'cot': {'blocks': 0, 'hits': 0}}
    seen_system_prompts = defaultdict(int)
    seen_user_prefixes = defaultdict(int)
    seen_tool_prefixes = defaultdict(int)
    seen_response_prefixes = defaultdict(int)
    seen_cot_prefixes = defaultdict(int)
    processed = 0
    for item in data[:n_requests * 2]:
        convs = item.get('conversations', [])
        if not convs:
            continue
        sys_prompt = random.choice(SYSTEM_PROMPTS)
        for msg in convs:
            from_field = msg.get('from', msg.get('role', ''))
            content = msg.get('value', msg.get('content', ''))
            if not content:
                continue
            if from_field == 'system':
                token_type = 'system_prompt'
            elif from_field in ('human', 'user'):
                token_type = 'user_query'
            elif from_field in ('gpt', 'assistant'):
                if '<think>' in content or 'chain of thought' in content.lower():
                    token_type = 'cot'
                else:
                    token_type = 'response'
            elif from_field in ('tool', 'function'):
                token_type = 'tool_output'
            else:
                continue
            prefix_hash = hash(content[:100])
            token_stats[token_type]['blocks'] += 1
            if token_type == 'system_prompt':
                sp_hash = hash(sys_prompt[:100])
                if sp_hash in seen_system_prompts:
                    token_stats[token_type]['hits'] += seen_system_prompts[sp_hash]
                seen_system_prompts[sp_hash] += 1
            elif token_type == 'user_query':
                if prefix_hash in seen_user_prefixes:
                    token_stats[token_type]['hits'] += 1
                seen_user_prefixes[prefix_hash] += 1
            elif token_type == 'tool_output':
                if prefix_hash in seen_tool_prefixes:
                    token_stats[token_type]['hits'] += 1
                seen_tool_prefixes[prefix_hash] += 1
            elif token_type == 'response':
                if prefix_hash in seen_response_prefixes:
                    token_stats[token_type]['hits'] += 1
                seen_response_prefixes[prefix_hash] += 1
            elif token_type == 'cot':
                if prefix_hash in seen_cot_prefixes:
                    token_stats[token_type]['hits'] += 1
                seen_cot_prefixes[prefix_hash] += 1
        processed += 1
        if processed >= n_requests:
            break
    total_requests = processed
    n_unique_sys = len(SYSTEM_PROMPTS)
    avg_sys_reuse = total_requests / n_unique_sys if n_unique_sys > 0 else 0
    results = {}
    for tt, stats in token_stats.items():
        blocks = max(stats['blocks'], 1)
        if tt == 'system_prompt':
            avg_hits = max(avg_sys_reuse - 1, 0) / (total_requests / blocks) if blocks > 0 else 0
            avg_hits = (total_requests - n_unique_sys) / blocks if blocks > 0 else 0
        else:
            avg_hits = stats['hits'] / blocks
        results[tt] = round(avg_hits, 4)
    return (results, processed)
print('=' * 70)
print('  Block-Level Reuse by Token Type Across 3 Datasets')
print('=' * 70)
all_results = {}
for ds_name, ds_path in DATASETS.items():
    print(f'\n  Loading {ds_name}...')
    import os
    if not os.path.exists(ds_path):
        print(f'    SKIP: file not found')
        continue
    with open(ds_path) as f:
        data = json.load(f)
    print(f'    Loaded {len(data)} conversations')
    results, n_processed = analyze_reuse(data, ds_name, N_REQUESTS)
    all_results[ds_name] = results
    print(f'    Processed: {n_processed} conversations')
    print(f'    Reuse rates:')
    for tt, val in results.items():
        print(f'      {tt:>15}: {val:.4f} avg hits/block')
print(f"\n{'=' * 70}")
print(f'  SUMMARY: Average Hits per Block by Token Type')
print(f"{'=' * 70}")
print(f"  {'Token Type':>15}  ", end='')
for ds in all_results:
    print(f'{ds:>15}', end='')
print()
print(f"  {'-' * 15}  ", end='')
for ds in all_results:
    print(f"{'-' * 15}", end='')
print()
for tt in ['system_prompt', 'user_query', 'tool_output', 'response', 'cot']:
    print(f'  {tt:>15}  ', end='')
    for ds in all_results:
        val = all_results[ds].get(tt, 0)
        print(f'{val:>15.4f}', end='')
    print()
print(f"{'=' * 70}")
outpath = '/workspace/block_reuse_3datasets.json'
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'  Saved to {outpath}')
