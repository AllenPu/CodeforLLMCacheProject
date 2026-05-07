import json, os, statistics
import numpy as np
DATASETS = {'ShareGPT': '/workspace/LPC/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json', 'LMSys': '/workspace/datasets/converted/lmsys_chat.json', 'Chatbot-Arena': '/workspace/datasets/converted/chatbot_arena.json'}
results = {}
for ds_name, ds_path in DATASETS.items():
    if not os.path.exists(ds_path):
        print(f'  SKIP {ds_name}: not found at {ds_path}')
        continue
    print(f"\n{'=' * 60}")
    print(f'  {ds_name}')
    print(f"{'=' * 60}")
    with open(ds_path) as f:
        data = json.load(f)
    chat_sessions = []
    agent_sessions = []
    chat_turn_counts = []
    agent_turn_counts = []
    chat_tokens_per_turn = []
    agent_tokens_per_turn = []
    chat_inter_turn_gaps = []
    agent_inter_turn_gaps = []
    for item in data:
        convs = item.get('conversations', [])
        if not convs or len(convs) < 2:
            continue
        roles = [msg.get('from', msg.get('role', '')) for msg in convs]
        is_agent = any((r in ('tool', 'function', 'function_call', 'tool_output', 'ipython', 'system_tool') for r in roles))
        user_turn_indices = []
        for i, msg in enumerate(convs):
            role = msg.get('from', msg.get('role', ''))
            if role in ('human', 'user'):
                user_turn_indices.append(i)
        if len(user_turn_indices) < 1:
            continue
        n_user_turns = len(user_turn_indices)
        turn_tokens = []
        for msg in convs:
            content = msg.get('value', msg.get('content', ''))
            tokens_approx = len(content.split()) * 1.3
            turn_tokens.append(tokens_approx)
        gaps = []
        for j in range(1, len(user_turn_indices)):
            prev_idx = user_turn_indices[j - 1]
            curr_idx = user_turn_indices[j]
            gap_tokens = sum(turn_tokens[prev_idx + 1:curr_idx])
            gaps.append(gap_tokens)
        if is_agent:
            agent_sessions.append(convs)
            agent_turn_counts.append(n_user_turns)
            agent_tokens_per_turn.extend(turn_tokens)
            agent_inter_turn_gaps.extend(gaps)
        else:
            chat_sessions.append(convs)
            chat_turn_counts.append(n_user_turns)
            chat_tokens_per_turn.extend(turn_tokens)
            chat_inter_turn_gaps.extend(gaps)

    def safe_stats(lst):
        if not lst:
            return {'n': 0, 'mean': 0, 'median': 0, 'std': 0, 'p25': 0, 'p75': 0}
        arr = np.array(lst)
        return {'n': len(arr), 'mean': float(np.mean(arr)), 'median': float(np.median(arr)), 'std': float(np.std(arr)), 'p25': float(np.percentile(arr, 25)), 'p75': float(np.percentile(arr, 75))}
    chat_gap_stats = safe_stats(chat_inter_turn_gaps)
    agent_gap_stats = safe_stats(agent_inter_turn_gaps)
    chat_turn_stats = safe_stats(chat_turn_counts)
    agent_turn_stats = safe_stats(agent_turn_counts)
    results[ds_name] = {'chat_sessions': len(chat_sessions), 'agent_sessions': len(agent_sessions), 'chat_turn_stats': chat_turn_stats, 'agent_turn_stats': agent_turn_stats, 'chat_gap_stats': chat_gap_stats, 'agent_gap_stats': agent_gap_stats}
    print(f'  Chat sessions:   {len(chat_sessions)}')
    print(f'  Agent sessions:  {len(agent_sessions)}')
    print(f'')
    print(f"  Chat turn count:  mean={chat_turn_stats['mean']:.2f}  median={chat_turn_stats['median']:.1f}")
    print(f"  Agent turn count: mean={agent_turn_stats['mean']:.2f}  median={agent_turn_stats['median']:.1f}")
    print(f'')
    print(f"  Chat inter-turn gap (tokens):  mean={chat_gap_stats['mean']:.1f}  median={chat_gap_stats['median']:.1f}  std={chat_gap_stats['std']:.1f}  n={chat_gap_stats['n']}")
    print(f"  Agent inter-turn gap (tokens): mean={agent_gap_stats['mean']:.1f}  median={agent_gap_stats['median']:.1f}  std={agent_gap_stats['std']:.1f}  n={agent_gap_stats['n']}")

    def bucket_dist(gaps, buckets=[0, 50, 100, 200, 500, 1000, 5000]):
        if not gaps:
            return {}
        arr = np.array(gaps)
        dist = {}
        for i in range(len(buckets) - 1):
            lo, hi = (buckets[i], buckets[i + 1])
            count = int(np.sum((arr >= lo) & (arr < hi)))
            pct = count / len(arr) * 100
            dist[f'{lo}-{hi}'] = {'count': count, 'pct': round(pct, 1)}
        count = int(np.sum(arr >= buckets[-1]))
        pct = count / len(arr) * 100
        dist[f'{buckets[-1]}+'] = {'count': count, 'pct': round(pct, 1)}
        return dist
    chat_dist = bucket_dist(chat_inter_turn_gaps)
    agent_dist = bucket_dist(agent_inter_turn_gaps)
    print(f'\n  Chat gap distribution (tokens):')
    for k, v in chat_dist.items():
        print(f"    {k:>10}: {v['count']:>8} ({v['pct']:.1f}%)")
    print(f'  Agent gap distribution (tokens):')
    for k, v in agent_dist.items():
        print(f"    {k:>10}: {v['count']:>8} ({v['pct']:.1f}%)")
    results[ds_name]['chat_gap_distribution'] = chat_dist
    results[ds_name]['agent_gap_distribution'] = agent_dist
print(f"\n{'=' * 60}")
print(f'  SUMMARY: Chat vs Agent Inter-Turn Gap (tokens)')
print(f"{'=' * 60}")
print(f"  {'Dataset':<18} {'Type':<8} {'N gaps':>8} {'Mean':>10} {'Median':>10} {'Std':>10}")
print(f"  {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
for ds_name, r in results.items():
    cg = r['chat_gap_stats']
    ag = r['agent_gap_stats']
    print(f"  {ds_name:<18} {'Chat':<8} {cg['n']:>8} {cg['mean']:>10.1f} {cg['median']:>10.1f} {cg['std']:>10.1f}")
    print(f"  {'':<18} {'Agent':<8} {ag['n']:>8} {ag['mean']:>10.1f} {ag['median']:>10.1f} {ag['std']:>10.1f}")
outpath = '/workspace/chat_vs_agent_analysis.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n  Saved to {outpath}')
