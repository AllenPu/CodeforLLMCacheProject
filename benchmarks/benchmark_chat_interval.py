import json, time, requests, sys, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
random.seed(42)
URL = 'http://localhost:8000/v1/chat/completions'
MODEL = '/workspace/models/Qwen2.5-1.5B-Instruct'
SYSTEM_PROMPTS = ['You are a helpful AI assistant that specializes in coding and software development. You write clean, efficient code and explain complex technical concepts clearly.', 'You are an expert travel planner who creates detailed itineraries for international trips. You consider budget, local customs, weather, and transportation options.', 'You are a professional chef and nutritionist who creates healthy, delicious recipes. You provide detailed ingredient lists, cooking instructions, and nutritional information.', 'You are a financial advisor who helps people manage their money wisely. You explain investment strategies, budgeting techniques, and tax optimization in simple terms.', 'You are a creative writing coach who helps people craft compelling stories. You provide feedback on narrative structure, character development, dialogue, and prose style.']
DATASET_PATHS = {'sharegpt': '/root/.etc/ShareGPT_V3_unfiltered_cleaned_split.json', 'lmsys': '/workspace/datasets/converted/lmsys_chat.json', 'chatbot_arena': '/workspace/datasets/converted/chatbot_arena.json'}
tag = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
lambda_chat = 5.0
dataset_name = 'sharegpt'
for i, arg in enumerate(sys.argv):
    if arg == '--lambda_chat' and i + 1 < len(sys.argv):
        lambda_chat = float(sys.argv[i + 1])
    if arg == '--dataset' and i + 1 < len(sys.argv):
        dataset_name = sys.argv[i + 1]
print(f"{'=' * 70}")
print(f'  Chat Interval Benchmark')
print(f'  Evictor: {tag}')
print(f'  Dataset: {dataset_name}')
print(f'  lambda_chat: {lambda_chat}s (mean think time)')
print(f"{'=' * 70}")
print('\n[1] Loading dataset...')
ds_path = DATASET_PATHS.get(dataset_name, DATASET_PATHS['sharegpt'])
with open(ds_path) as f:
    data = json.load(f)
sessions = []
MAX_SESSIONS = 500
MAX_TURNS = 4
for item in data:
    convs = item.get('conversations', [])
    if not convs:
        continue
    turns = []
    current_messages = []
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    current_messages.append({'role': 'system', 'content': sys_prompt})
    for msg in convs:
        from_field = msg.get('from', msg.get('role', ''))
        content = msg.get('value', msg.get('content', ''))[:300]
        if from_field in ('human', 'user'):
            role = 'user'
        elif from_field in ('gpt', 'assistant'):
            role = 'assistant'
        else:
            continue
        if not content:
            continue
        current_messages.append({'role': role, 'content': content})
        if role == 'assistant':
            turns.append([m.copy() for m in current_messages])
    if len(turns) >= 2:
        session_requests = []
        for t in range(min(len(turns), MAX_TURNS)):
            msgs = turns[t]
            if msgs[-1]['role'] != 'user':
                msgs.append({'role': 'user', 'content': 'Continue.'})
            session_requests.append(msgs)
        sessions.append(session_requests)
    if len(sessions) >= MAX_SESSIONS:
        break
print(f'  Sessions: {len(sessions)} (multi-turn conversations)')
print(f'  Total requests: {sum((len(s) for s in sessions))}')
avg_turns = sum((len(s) for s in sessions)) / max(len(sessions), 1)
print(f'  Avg turns/session: {avg_turns:.1f}')
print(f'\n[2] Running benchmark (lambda_chat={lambda_chat}s)...')
ttfts_by_turn = {}
all_ttfts = []
errors = 0
completed_sessions = 0
lock = threading.Lock()

def run_session(session_requests):
    global errors, completed_sessions
    session_ttfts = []
    for turn_idx, messages in enumerate(session_requests):
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
            if ttft is None:
                ttft = time.time() - t0
            session_ttfts.append((turn_idx, ttft))
            if turn_idx < len(session_requests) - 1:
                think_time = random.expovariate(1.0 / lambda_chat)
                think_time = min(think_time, lambda_chat * 3)
                time.sleep(think_time)
        except Exception as e:
            with lock:
                errors += 1
    with lock:
        completed_sessions += 1
        for turn_idx, ttft in session_ttfts:
            all_ttfts.append(ttft)
            if turn_idx not in ttfts_by_turn:
                ttfts_by_turn[turn_idx] = []
            ttfts_by_turn[turn_idx].append(ttft)
    return session_ttfts
N_CONCURRENT = 32
start = time.time()
with ThreadPoolExecutor(max_workers=N_CONCURRENT) as executor:
    futures = []
    for i, session in enumerate(sessions):
        futures.append(executor.submit(run_session, session))
        time.sleep(0.05)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f'    Launched {i + 1}/{len(sessions)} sessions, {elapsed:.0f}s')
    for f in as_completed(futures):
        f.result()
elapsed = time.time() - start
all_ttfts.sort()
n = len(all_ttfts)
if n > 0:
    mean_ttft = sum(all_ttfts) / n
    p50 = all_ttfts[n // 2]
    p90 = all_ttfts[int(n * 0.9)]
    p99 = all_ttfts[int(n * 0.99)]
else:
    mean_ttft = p50 = p90 = p99 = 0
print(f"\n{'=' * 70}")
print(f'  RESULTS (tag={tag}, dataset={dataset_name}, lambda_chat={lambda_chat}s)')
print(f"{'=' * 70}")
print(f'  Sessions:  {completed_sessions}')
print(f'  Requests:  {n} total, {errors} errors')
print(f'  Duration:  {elapsed:.0f}s')
print(f'  Overall:   Mean={mean_ttft * 1000:.1f}ms  P50={p50 * 1000:.1f}ms  P90={p90 * 1000:.1f}ms  P99={p99 * 1000:.1f}ms')
print(f'\n  Per-turn TTFT breakdown:')
print(f"  {'Turn':>6} {'Count':>8} {'Mean':>12} {'P50':>10}")
print(f"  {'-' * 6} {'-' * 8} {'-' * 12} {'-' * 10}")
for turn_idx in sorted(ttfts_by_turn.keys()):
    t_list = sorted(ttfts_by_turn[turn_idx])
    t_n = len(t_list)
    t_mean = sum(t_list) / t_n
    t_p50 = t_list[t_n // 2]
    print(f'  {turn_idx:>6} {t_n:>8} {t_mean * 1000:>10.1f}ms {t_p50 * 1000:>8.1f}ms')
print(f"{'=' * 70}")
result = {'tag': tag, 'dataset': dataset_name, 'lambda_chat': lambda_chat, 'n': n, 'errors': errors, 'mean': mean_ttft, 'p50': p50, 'p90': p90, 'p99': p99, 'ttfts': all_ttfts, 'per_turn': {str(k): {'count': len(v), 'mean': sum(v) / len(v), 'p50': sorted(v)[len(v) // 2]} for k, v in ttfts_by_turn.items()}, 'sessions': completed_sessions, 'elapsed': elapsed}
outpath = f'/workspace/ttft_interval_{dataset_name}_{tag}_lambda{int(lambda_chat)}.json'
with open(outpath, 'w') as f:
    json.dump(result, f)
print(f'  Saved to {outpath}')
