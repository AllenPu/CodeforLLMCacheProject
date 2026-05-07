import json, os
DATASETS = {'ShareGPT': '/workspace/LPC/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json', 'LMSys': '/workspace/datasets/converted/lmsys_chat.json', 'Chatbot-Arena': '/workspace/datasets/converted/chatbot_arena.json'}
ALT_SHAREGPT = '/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json'
results = {}
for ds_name, ds_path in DATASETS.items():
    if not os.path.exists(ds_path):
        if ds_name == 'ShareGPT' and os.path.exists(ALT_SHAREGPT):
            ds_path = ALT_SHAREGPT
        else:
            print(f'  SKIP {ds_name}: not found')
            continue
    print(f"\n{'=' * 60}")
    print(f'  {ds_name}')
    print(f"{'=' * 60}")
    with open(ds_path) as f:
        data = json.load(f)
    total = 0
    single = 0
    multi = 0
    turn_counts = []
    turn_dist = {1: 0, 2: 0, 3: 0, 4: 0, '5+': 0}
    for item in data:
        convs = item.get('conversations', [])
        if not convs:
            continue
        user_turns = sum((1 for msg in convs if msg.get('from', msg.get('role', '')) in ('human', 'user')))
        if user_turns == 0:
            continue
        total += 1
        turn_counts.append(user_turns)
        if user_turns == 1:
            single += 1
            turn_dist[1] += 1
        else:
            multi += 1
            if user_turns == 2:
                turn_dist[2] += 1
            elif user_turns == 3:
                turn_dist[3] += 1
            elif user_turns == 4:
                turn_dist[4] += 1
            else:
                turn_dist['5+'] += 1
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    multi_pct = multi / total * 100 if total else 0
    single_pct = single / total * 100 if total else 0
    total_requests = sum(turn_counts)
    followup_requests = total_requests - total
    followup_ratio = followup_requests / total_requests * 100 if total_requests else 0
    results[ds_name] = {'total_conversations': total, 'single_turn': single, 'single_turn_pct': single_pct, 'multi_turn': multi, 'multi_turn_pct': multi_pct, 'avg_user_turns': avg_turns, 'total_requests': total_requests, 'followup_requests': followup_requests, 'followup_ratio': followup_ratio, 'turn_distribution': turn_dist}
    print(f'  Total conversations:  {total}')
    print(f'  Single-turn:          {single} ({single_pct:.1f}%)')
    print(f'  Multi-turn:           {multi} ({multi_pct:.1f}%)')
    print(f'  Avg user turns:       {avg_turns:.2f}')
    print(f'  Total requests:       {total_requests}')
    print(f'  Follow-up requests:   {followup_requests} ({followup_ratio:.1f}%)')
    print(f'  Turn distribution:')
    for k, v in turn_dist.items():
        pct = v / total * 100 if total else 0
        print(f'    {k} turn(s): {v} ({pct:.1f}%)')
print(f"\n{'=' * 60}")
print(f'  SUMMARY')
print(f"{'=' * 60}")
print(f"  {'Dataset':<18} {'Total':>8} {'Single':>8} {'Multi':>8} {'Multi%':>8} {'AvgTurn':>8} {'FollowUp%':>10}")
print(f"  {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
for ds_name, r in results.items():
    print(f"  {ds_name:<18} {r['total_conversations']:>8} {r['single_turn']:>8} {r['multi_turn']:>8} {r['multi_turn_pct']:>7.1f}% {r['avg_user_turns']:>7.2f} {r['followup_ratio']:>9.1f}%")
outpath = '/workspace/multi_turn_analysis.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n  Saved to {outpath}')
