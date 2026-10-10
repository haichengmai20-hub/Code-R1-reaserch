from pathlib import Path
import ast
import copy
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import time

ROOT = Path('D:/RL及harness/code-r1')
OUT = Path(__file__).parent
IDS = ['find-the-longest-equal-subarray', 'count-zero-request-servers',
       'relocate-marbles', 'sum-of-matrix-after-queries', 'maximum-sum-queries']
files = sorted((ROOT / 'LeetCodeDataset/data').glob('*.jsonl'))
datasets = {p.name: [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] for p in files}
tasks = {t['task_id']: t for t in datasets['LeetCodeDataset-v1-test-problems.jsonl'] if t['task_id'] in IDS}
report = {'python': sys.version, 'datasets': {}, 'tasks': [], 'differential_checks': {}, 'reward_probes': {}}
report['protected_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
report['original_doc_hash'] = hashlib.sha256((OUT/'original.md').read_bytes()).hexdigest()
for name, rows in datasets.items():
    report['datasets'][name] = {'rows': len(rows), 'selected_ids': [t['task_id'] for t in rows if t['task_id'] in IDS]}

def load_fn(t, code=None, prelude=True):
    env = {}
    # Only the five inspected, stdlib reference snippets and example assertions run here.
    exec((t['prompt'] if prelude else '') + (code or t['completion']), env)
    return eval(t['entry_point'], env), env

functions = {}
for task_id in IDS:
    t = tasks[task_id]
    for text in [t['prompt']+t['completion'], t['test']]:
        ast.parse(text)
    fn, env = load_fn(t)
    exec(t['test'], env)
    env['check'](fn)
    functions[task_id] = fn
    item = {'task_id': task_id, 'question_id': t['meta']['question_id'], 'split': t['meta']['split'],
            'asserts': sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(t['test']))),
            'query_has_constraints_heading': 'Constraints' in t['meta']['query'], 'reference_seed_tests': 'pass'}
    try:
        fn0, env0 = load_fn(t, prelude=False)
        exec(t['test'], env0)
        env0['check'](fn0)
        item['without_prelude'] = 'pass on this Python version'
    except Exception as e:
        item['without_prelude'] = f'{type(e).__name__}: {e}'
    if task_id != 'maximum-sum-queries':
        code = max(re.findall(r'```python\n(.*?)```', t['meta']['response'], re.S), key=len)
        f1, e1 = load_fn(t, code=code)
        exec(t['test'], e1)
        e1['check'](f1)
        item['stored_response_seed_tests'] = 'pass'
    else:
        code = max(re.findall(r'```python\n(.*?)```', t['meta']['response'], re.S), key=len)
        ast.parse(code)
        item['stored_response_seed_tests'] = 'not executed: sortedcontainers dependency not installed for this audit'
    report['tasks'].append(item)

def longest_oracle(nums, k):
    ans = 0
    for left in range(len(nums)):
        cnt = {}
        for right in range(left, len(nums)):
            cnt[nums[right]] = cnt.get(nums[right], 0) + 1
            mx = max(cnt.values())
            if right-left+1-mx <= k:
                ans = max(ans, mx)
    return ans

def servers_oracle(n, logs, x, queries):
    return [n-len({s for s, tm in logs if q-x <= tm <= q}) for q in queries]

def marbles_oracle(nums, moveFrom, moveTo):
    a = list(nums)
    for f, t in zip(moveFrom, moveTo):
        a = [t if v == f else v for v in a]
    return sorted(set(a))

def matrix_oracle(n, queries):
    grid = [[0]*n for _ in range(n)]
    for typ, idx, val in queries:
        if typ == 0:
            grid[idx] = [val]*n
        else:
            for row in grid:
                row[idx] = val
    return sum(map(sum, grid))

def maxsum_oracle(nums1, nums2, queries):
    return [max((a+b for a,b in zip(nums1,nums2) if a >= x and b >= y), default=-1) for x,y in queries]

oracles = dict(zip(IDS, [longest_oracle, servers_oracle, marbles_oracle, matrix_oracle, maxsum_oracle]))
def compare(task_id, kwargs):
    expected = oracles[task_id](**copy.deepcopy(kwargs))
    actual = functions[task_id](**copy.deepcopy(kwargs))
    assert actual == expected, (task_id, kwargs, expected, actual)

count = 0
for n in range(1, 7):
    for nums in itertools.product(range(1, min(n, 3)+1), repeat=n):
        for k in range(n+1):
            compare(IDS[0], {'nums': list(nums), 'k': k})
            count += 1
report['differential_checks']['longest_exhaustive'] = count
rng = random.Random(20261003)
for task_id in IDS:
    for _ in range(200):
        n = rng.randint(1, 8)
        if task_id == IDS[0]:
            kw = {'nums': [rng.randint(1,n) for _ in range(n)], 'k': rng.randint(0,n)}
        elif task_id == IDS[1]:
            x = rng.randint(1,5)
            kw = {'n': n, 'logs': [[rng.randint(1,n),rng.randint(1,15)] for _ in range(rng.randint(1,10))],
                  'x': x, 'queries': [rng.randint(x+1,18) for _ in range(rng.randint(1,8))]}
        elif task_id == IDS[2]:
            nums = [rng.randint(1,8) for _ in range(n)]
            occupied = set(nums)
            fs, ts = [], []
            for _ in range(rng.randint(1,8)):
                f, t = rng.choice(sorted(occupied)), rng.randint(1,8)
                fs.append(f); ts.append(t)
                occupied.remove(f); occupied.add(t)
            kw = {'nums': nums, 'moveFrom': fs, 'moveTo': ts}
        elif task_id == IDS[3]:
            kw = {'n': n, 'queries': [[rng.randint(0,1),rng.randrange(n),rng.randint(0,10)] for _ in range(rng.randint(1,10))]}
        else:
            kw = {'nums1': [rng.randint(1,20) for _ in range(n)], 'nums2': [rng.randint(1,20) for _ in range(n)],
                  'queries': [[rng.randint(1,22),rng.randint(1,22)] for _ in range(rng.randint(1,8))]}
        compare(task_id, kw)
    report['differential_checks'][task_id+'_random'] = 200

edge = {'n':2, 'logs':[[1,1]], 'x':1, 'queries':[2,3]}
report['servers_boundary'] = {'expected': servers_oracle(**copy.deepcopy(edge)),
                              'actual': functions[IDS[1]](**copy.deepcopy(edge))}
pair_list = sorted([(1,101),(20,21)])
report['tuple_bug_probe'] = {'expected': maxsum_oracle([100,1],[1,20],[[1,1]]),
                            'stored_expression_result': [max(pair_list)[1]],
                            'scope': 'reproduced expression semantics; not full SortedList program execution'}

# Reuse the exact parser and scoring function from source; replace only the
# Linux backend with a local subprocess for controlled, inspected probes.
src = (ROOT/'verl/utils/reward_score/coder1/__init__.py').read_text(encoding='utf-8')
tree = ast.parse(src)
selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in
             {'validate_response_structure','try_extract_solution','extract_code_from_string','_compute_score'})
            or (isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE_PATTERN' for t in n.targets))]
def controlled_exec(code):
    p = subprocess.run([sys.executable,'-I','-c',code], capture_output=True, text=True, timeout=10)
    return p.returncode == 0, p.stdout+p.stderr
env = {'re':re, 'json':json, 'time':time, 'Tuple':tuple, 'Optional':__import__('typing').Optional,
       '_MAX_CHAR_DISPLAY':2048, 'code_exec':controlled_exec}
exec(compile(ast.Module(body=selected, type_ignores=[]),'<audited scoring functions>','exec'),env)
t = tasks[IDS[1]]
gt = json.dumps({'functional': t['test']+'\n\ncheck('+t['entry_point']+')'})
def formatted(code):
    return '<think>audit</think><answer>\n```python\n'+code+'\n```\n</answer>'
probes = {'bad_format': 'not a code block',
          'correct': formatted(t['prompt']+t['completion']),
          'wrong_answer': formatted('class Solution:\n    def countServers(self, n, logs, x, queries):\n        return [-1]*len(queries)\n'),
          'early_exit': formatted('import sys\nsys.exit(0)')}
for name, response in probes.items():
    score, log = env['_compute_score'](response,gt,{'prompt':'controlled probe'})
    report['reward_probes'][name] = score
report['reward_probe_scope'] = 'Exact source parser/_compute_score, local subprocess exit-code contract; no Firejail/Docker/backend audit'
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
