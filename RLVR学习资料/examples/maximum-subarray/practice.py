"""Student data/GT exercises. Fixed reviewed programs only; no arbitrary-code CLI."""
import argparse
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixed_demo', HERE / 'run_demo.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MAX_SMALL_INPUT = 64
MAX_CASES = 64
CANDIDATE_IDS = ('reference', 'correct_prefix', 'wrong_empty', 'wrong_sum', 'bad_format', 'early_exit')


def run_directory(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Use a simple run name, e.g. data-01; paths are not accepted.')
    parent = (HERE / 'practice_runs').resolve()
    target = (parent / name).resolve()
    if not parent.is_relative_to(HERE) or not target.is_relative_to(parent):
        raise ValueError('Exercise paths must remain inside practice_runs.')
    return target


def safe_path(folder, filename):
    target = (folder / filename).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise ValueError('Exercise file resolves outside its run directory.')
    return target


def save(folder, filename, value):
    folder.mkdir(parents=True, exist_ok=True)
    path = safe_path(folder, filename)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(folder, filename):
    path = safe_path(folder, filename)
    if path.stat().st_size > 2_000_000:
        raise ValueError('Exercise JSON exceeds the data budget.')
    return json.loads(path.read_text(encoding='utf-8'))


def attempt(folder):
    for number in range(1, 10000):
        result = safe_path(folder, 'attempt-%03d' % number)
        if not result.exists():
            result.mkdir(parents=True)
            return result
    raise ValueError('Choose a new run name after 9999 attempts.')


def initial_cases(mode):
    base.check_pinned_sources()
    raw, _ = base.load_raw()
    result = base.extract_seed_cases(raw)
    if mode == 'expanded':
        result += [c for c in base.diagnostic_cases() if c['origin']['kind'] != 'intentional_negative_control']
    elif mode != 'seed':
        raise ValueError('Mode must be seed or expanded.')
    return copy.deepcopy(result)


def audit_cases(proposals):
    """No fixed 8/7 counts. Preserve each accepted pair when a sibling is bad."""
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= MAX_CASES:
        raise ValueError('Provide 1..64 case records.')
    seen = set()
    for item in proposals:
        if not isinstance(item, dict) or not {'case_id', 'input', 'expected'} <= item.keys():
            raise ValueError('Each record needs case_id, input and expected.')
        cid = item['case_id']
        if not isinstance(cid, str) or not cid or len(cid) > 100 or cid in seen:
            raise ValueError('Case IDs must be unique nonempty strings, at most 100 characters.')
        seen.add(cid)
        if not isinstance(item['input'], dict) or set(item['input']) != {'nums'}:
            raise ValueError('This exercise accepts only the maxSubArray nums interface.')
        if 'input_sha256' in item and item['input_sha256'] != base.digest(base.canonical(item['input'])):
            raise ValueError('Input/hash mismatch: ' + cid)
    accepted, quarantined, pending = [], [], []
    verified_by_input = {}
    for item in proposals:
        case = copy.deepcopy(item)
        case.setdefault('origin', 'learner; record your derivation in prediction.json')
        case['input_sha256'] = base.digest(base.canonical(case['input']))
        nums = case['input']['nums']
        reason = base.legality(nums) if isinstance(nums, list) else 'nums_must_be_list'
        if reason or type(case['expected']) is not int:
            case.update(status='quarantined', reasons=[reason or 'expected_must_be_int'])
            quarantined.append(case)
            continue
        if len(nums) > MAX_SMALL_INPUT:
            case.update(status='pending', reason='outside_small_oracle_budget')
            pending.append(case)
            continue
        brute, witness = base.brute_force(nums)
        dynamic = base.kadane(nums)
        if brute != dynamic:
            raise ValueError('Independent methods disagree; stop and review the specification.')
        case['verification'] = {'enumeration': brute, 'witness': witness, 'independent_kadane': dynamic}
        if case['expected'] != brute:
            case.update(status='quarantined', reasons=['expected_mismatch'])
            if case['input_sha256'] in verified_by_input:
                case['correct_same_input_case_id'] = verified_by_input[case['input_sha256']]
            quarantined.append(case)
        else:
            case.update(status='accepted', reasons=[])
            accepted.append(case)
            verified_by_input[case['input_sha256']] = case['case_id']
    if accepted:
        reference = base.collect_outputs(base.REFERENCE, [c['input']['nums'] for c in accepted])
        if reference != [c['expected'] for c in accepted]:
            raise ValueError('Reviewed reference disagrees with independent answers.')
    return {'accepted': accepted, 'quarantined': quarantined, 'pending': pending,
            'scope': 'Small-input per-pair checks; not universal correctness or training admission.'}


def functional_gt(accepted):
    return base.generate_functional(accepted)


def score_cases(accepted, format_ok=True):
    gt = functional_gt(accepted)
    candidates = {'reference': base.REFERENCE, 'correct_prefix': base.CORRECT_PREFIX,
                  'wrong_empty': base.WRONG_EMPTY, 'wrong_sum': base.WRONG_SUM, 'early_exit': base.EARLY_EXIT}
    allowed = {code.rstrip() + '\n' + gt for code in candidates.values()}
    scorer = base.load_original_scorer(allowed)
    records = []
    for name in CANDIDATE_IDS:
        code = candidates.get(name, base.REFERENCE)
        response = base.response(code) if format_ok and name != 'bad_format' else '```python\n' + code.rstrip() + '\n```'
        reward, log = scorer['_compute_score'](response, base.canonical({'functional': gt}),
                                              {'prompt': 'Controlled student maxSubArray exercise'})
        values = base.collect_outputs(code, [c['input']['nums'] for c in accepted]) if format_ok and name != 'bad_format' else None
        complete = isinstance(values, list) and len(values) == len(accepted)
        observations = [] if not complete else [
            {'case_id': c['case_id'], 'expected': c['expected'], 'actual': value,
             'equal': type(value) is int and value == c['expected']}
            for c, value in zip(accepted, values)]
        records.append({'candidate_id': name, 'reward': reward, 'source_log': log,
                        'audit_completed_case_count': len(observations),
                        'correctness_pass': complete and all(x['equal'] for x in observations),
                        'observations': observations})
    return records


def two_sum_valid(nums, target, answer):
    return (isinstance(answer, list) and len(answer) == 2 and
            all(type(i) is int and 0 <= i < len(nums) for i in answer) and
            answer[0] != answer[1] and nums[answer[0]] + nums[answer[1]] == target)


def pass_at_k(n, c, k):
    if any(type(v) is not int for v in (n, c, k)) or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Need 0<=c<=n and 1<=k<=n, all integers.')
    return 1.0 if n-c < k else 1-math.comb(n-c, k)/math.comb(n, k)


def diagnose_packet(packet):
    """Inspect data, never execute learner-supplied GT or code."""
    layers = []
    if packet['entry_point'] != 'Solution().maxSubArray': layers.append('Task')
    nums = packet['input']['nums']
    if base.legality(nums) is not None or len(nums) > MAX_SMALL_INPUT:
        layers.append('Tests')
    elif type(packet['expected']) is not int or base.brute_force(nums)[0] != packet['expected']:
        layers.append('Tests')
    if type(packet['gt_expected']) is not int or packet['gt_expected'] != packet['expected']: layers.append('GT')
    values=packet.get('actual_values', [])
    completion=(type(packet['observed_cases']) is int and type(packet['required_cases']) is int and
                packet['required_cases'] > 0 and packet['observed_cases']==packet['required_cases'] and
                isinstance(values,list) and len(values)==packet['required_cases'])
    # A bad expected must first be repaired; it is not evidence that the candidate is wrong.
    actual_correct=completion and all(type(v) is int and v==packet['expected'] for v in values)
    if type(packet['claims_correctness']) is not bool or (packet['claims_correctness'] and
            (not completion or packet['scorer_reward']!=1.1 or ('Tests' not in layers and not actual_correct))):
        layers.append('Verifier')
    try:
        truth = pass_at_k(packet['n'], packet['c'], packet['k'])
        if not math.isclose(truth, packet['reported_pass_at_k'], abs_tol=1e-9): layers.append('Evaluation')
    except (ValueError, TypeError):
        layers.append('Evaluation')
    return layers or ['none']


def blind_packets(seed):
    rng = random.Random(seed)
    packets=[]
    for _ in range(8):
        nums=[rng.randint(-7,7) for _ in range(rng.randint(2,6))]
        truth=base.brute_force(nums)[0]
        packets.append({'input':{'nums':nums},'expected':truth,'gt_expected':truth,
                        'entry_point':'Solution().maxSubArray','scorer_reward':1.1,
                        'observed_cases':1,'required_cases':1,'claims_correctness':True,'actual_values':[truth],
                        'n':4,'c':2,'k':2,'reported_pass_at_k':5/6})
    packets[1]['expected'] += rng.randint(2,7)
    packets[1]['gt_expected'] = packets[1]['expected']
    packets[2]['gt_expected'] += rng.randint(2,7)
    packets[3]['observed_cases'] = 0
    packets[3]['actual_values'] = []
    packets[4].update(n=1, c=1, k=4, reported_pass_at_k=1.)
    packets[5]['entry_point'] = 'Solution().maxSubarray'
    packets[6]['gt_expected'] += rng.randint(2,7)
    packets[6].update(n=1,c=1,k=4,reported_pass_at_k=1.)
    rng.shuffle(packets)
    for i, packet in enumerate(packets): packet['record_id'] = 'record-%02d' % (i+1)
    return packets


def transfer_material():
    base.check_pinned_sources()
    for number, line in enumerate(base.DATASET.read_text(encoding='utf-8').splitlines(), 1):
        raw = json.loads(line)
        if raw['task_id'] == 'two-sum': break
    else: raise ValueError('Pinned two-sum task missing.')
    tests = []
    for node in ast.walk(ast.parse(raw['test'])):
        if isinstance(node, ast.Assert):
            tests.append({'input': {k.arg: ast.literal_eval(k.value) for k in node.test.left.keywords},
                          'expected': ast.literal_eval(node.test.comparators[0])})
    code = """class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, value in enumerate(nums):
            if target-value in seen:
                return [seen[target-value], i]
            seen[value] = i
"""
    return {'raw_task': raw, 'source_line_1_based': number, 'raw_record_sha256': base.digest(line),
            'tests': tests, 'candidate_sources': {
                'A': code, 'B': code.replace('return [seen[target-value], i]', 'return [i, seen[target-value]]'),
                'C': code.replace('            if target-value in seen:',
                                  '            seen[value] = i\n            if target-value in seen:')}}


def transfer_results(material):
    records = {}
    calls = [c['input'] for c in material['tests']]
    for cid, code in material['candidate_sources'].items():
        # Code is regenerated from reviewed constants, never read back from student JSON.
        source = code + '\nimport json\nprint(json.dumps([Solution().twoSum(**x) for x in ' + repr(calls) + ']))\n'
        success, output = base.controlled_exec(source, {source})
        if not success: raise ValueError(output)
        actual = json.loads(output)
        records[cid] = {'legacy_pass': all(v == t['expected'] for v,t in zip(actual,material['tests'])),
                        'semantic_pass': all(two_sum_valid(t['input']['nums'],t['input']['target'],v)
                                             for v,t in zip(actual,material['tests'])), 'actual': actual}
    return records


def transfer_semantic_gt(material):
    checks = ['def _valid(nums, target, answer):',
              '    return (isinstance(answer, list) and len(answer) == 2 and',
              '            all(type(i) is int and 0 <= i < len(nums) for i in answer) and',
              '            answer[0] != answer[1] and nums[answer[0]] + nums[answer[1]] == target)',
              '', 'def check(candidate):']
    for i, case in enumerate(material['tests'],1):
        nums=case['input']['nums'];target=case['input']['target']
        checks.append('    assert _valid(%r, %r, candidate(nums=%r, target=%r)), %r' %
                      (nums,target,nums,target,'two-sum-seed-%d'%i))
    checks.extend(['','check(Solution().twoSum)',''])
    return '\n'.join(checks)


def transfer_gt_regression(material, results):
    legacy=material['raw_task']['test']+'\n\ncheck(Solution().twoSum)'
    semantic=transfer_semantic_gt(material)
    allowed={code.rstrip()+'\n'+gt for code in material['candidate_sources'].values() for gt in (legacy,semantic)}
    scorer=base.load_original_scorer(allowed)
    for cid,code in material['candidate_sources'].items():
        for name,gt in (('legacy',legacy),('semantic',semantic)):
            reward,_=scorer['_compute_score'](base.response(code),base.canonical({'functional':gt}),
                                             {'prompt':material['raw_task']['meta']['query']})
            if (reward==1.1)!=results[cid][name+'_pass']:
                raise ValueError('Controller comparison and assembled GT disagree for '+cid)
            results[cid][name+'_reward']=reward
    return semantic


def assert_transfer_material(submitted, original):
    if base.digest(base.canonical(submitted)) != base.digest(base.canonical(original)):
        raise ValueError('Transfer material changed, including numeric types. Keep the pinned task fixed.')


def student_row(accepted):
    raw, source = base.load_raw()
    gt = functional_gt(accepted)
    version = 'learner-' + base.digest(gt)[:12]
    prompt = raw['meta']['query']+'\n\n'+base.ACTOR_PROMPT_APPEND
    return {'data_source': 'code', 'ability': 'coding',
            'prompt': [{'role':'system','content':base.extract_system_prompt()},
                       {'role':'user','content':prompt}],
            'reward_model': {'style':'rule','ground_truth':base.canonical({'functional':gt})},
            'extra_info': {'task_id':raw['task_id'],'entry_point':raw['entry_point'],
                          'index':source['source']['line_1_based']-1,'prompt':prompt,
                          'dataset':'LeetCodeDataset','reference':raw['completion'],
                          'original_corpus_split':raw['meta']['split'],'split':'diagnostic',
                          'source':source['source'],'test_version':version,'train_ready':False,
                          'case_index':[{k:c[k] for k in ('case_id','input','expected','input_sha256','origin')}
                                        for c in accepted]}}


def require_predictions(prediction, keys):
    for key in keys:
        if key not in prediction or prediction[key] is None:
            raise ValueError('Fill prediction.json before running: ' + key)
    if not isinstance(prediction.get('why'),str) or not prediction['why'].strip():
        raise ValueError('Write your reasoning in prediction.json: why')


def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init','check','score','transfer-init','transfer-grade','blind-init','blind-grade'])
    parser.add_argument('--run', required=True, help='Separate exercise run, e.g. data-01')
    parser.add_argument('--mode', choices=['seed','expanded'], default='seed')
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--bad-format', action='store_true', help='Remove answer tags for a format-only contrast.')
    args=parser.parse_args(); folder=run_directory(args.run); status=0
    before=base.readonly_source_fingerprints()
    if args.command.endswith('init') or args.command=='init':
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Run already exists. Choose another --run; existing work is preserved.')
        if args.command=='init':
            save(folder,'cases.json',initial_cases(args.mode))
            save(folder,'prediction.json',{'accepted':None,'quarantined':None,'pending':None,'wrong_empty_reward':None,'why':''})
        elif args.command=='transfer-init':
            material=transfer_material();save(folder,'material.json',material)
            save(folder,'init_manifest.json',{'kind':'transfer','material_sha256':base.digest(base.canonical(material))})
            save(folder,'prediction.json',{**{cid:{'legacy_pass':None,'semantic_pass':None} for cid in ('A','B','C')},'why':''})
        else:
            packets=blind_packets(args.seed);save(folder,'packets.json',packets)
            save(folder,'init_manifest.json',{'kind':'blind','packets_sha256':base.digest(base.canonical(packets)),'seed':args.seed})
            save(folder,'prediction.json',{'layers':{p['record_id']:None for p in blind_packets(args.seed)},'why':''})
        print('Initialized: '+str(folder));print('Edit data and prediction.json, then use the corresponding check/grade command.')
    elif args.command in ('check','score'):
        prediction=read(folder,'prediction.json');require_predictions(prediction,['accepted','quarantined','pending'])
        if args.command=='score': require_predictions(prediction,['wrong_empty_reward'])
        cases=read(folder,'cases.json'); audit=audit_cases(cases); destination=attempt(folder)
        save(destination,'submitted_prediction.json',prediction);save(destination,'submitted_cases.json',cases)
        save(destination,'cases_audit.json',audit)
        counts={k:len(audit[k]) for k in ('accepted','quarantined','pending')}
        report={'counts':counts,'data_prediction_matches':all(type(prediction[k]) is int and prediction[k]==counts[k] for k in counts),
                'input_sha256':base.digest(base.canonical(cases)),'stage':args.command,
                'scope':'Student data changes / reviewed fixed programs; no model or general secure executor.'}
        if audit['accepted']:
            row=student_row(audit['accepted']);save(destination,'prepared_task.json',row)
            path=safe_path(destination,'data_pool.jsonl');path.write_text(base.canonical(row)+'\n',encoding='utf-8')
            assert json.loads(path.read_text(encoding='utf-8'))==row
            report['test_version']=row['extra_info']['test_version']
        if args.command=='score':
            if not audit['accepted']: raise ValueError('No accepted tests; GT and scoring are refused.')
            report['candidates']=score_cases(audit['accepted'],not args.bad_format)
            observed=next(x['reward'] for x in report['candidates'] if x['candidate_id']=='wrong_empty')
            report['reward_prediction_matches']=type(prediction['wrong_empty_reward']) in (int,float) and math.isclose(prediction['wrong_empty_reward'],observed,abs_tol=1e-9)
            _,_,adv=base.normalized([x['reward'] for x in report['candidates'][:4]])
            report['teaching_advantages']=adv
        save(destination,'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
        status=0 if report['data_prediction_matches'] and report.get('reward_prediction_matches',True) else 1
        print('Preserved attempt: '+str(destination))
    elif args.command=='transfer-grade':
        prediction=read(folder,'prediction.json');require_predictions(prediction,['A','B','C'])
        if any(type(prediction[c].get(k)) is not bool for c in ('A','B','C') for k in ('legacy_pass','semantic_pass')):
            raise ValueError('Fill all candidate predictions with JSON true/false.')
        material=transfer_material()
        submitted_material=read(folder,'material.json');assert_transfer_material(submitted_material,material)
        if read(folder,'init_manifest.json')['material_sha256']!=base.digest(base.canonical(submitted_material)):
            raise ValueError('Transfer material changed. Grading must use the same pinned task/candidates you inspected.')
        results=transfer_results(material);semantic=transfer_gt_regression(material,results);destination=attempt(folder)
        save(destination,'submitted_prediction.json',prediction)
        save(destination,'semantic_ground_truth.json',{'functional':semantic})
        raw=material['raw_task'];prompt=raw['meta']['query']
        save(destination,'prepared_task.json',{'data_source':'code','ability':'coding',
             'prompt':[{'role':'system','content':base.extract_system_prompt()},{'role':'user','content':prompt}],
             'reward_model':{'style':'rule','ground_truth':base.canonical({'functional':semantic})},
             'extra_info':{'task_id':'two-sum','entry_point':raw['entry_point'],'index':material['source_line_1_based']-1,
                           'prompt':prompt,'reference':raw['completion'],'dataset':'LeetCodeDataset','split':'diagnostic',
                           'test_version':'two-sum-semantic-diagnostic-v1','train_ready':False,
                           'source_line_1_based':material['source_line_1_based'],'raw_record_sha256':material['raw_record_sha256']}})
        matches=all(prediction[c][k]==results[c][k] for c in results for k in ('legacy_pass','semantic_pass'))
        save(destination,'report.json',{'prediction_matches':matches,'results':results,
                                     'source_line_1_based':material['source_line_1_based'],
                                     'scope':'Two-sum finite seed comparator exercise; not model evaluation.'})
        print(json.dumps({'prediction_matches':matches,'results':results},ensure_ascii=False,indent=2))
        status=0 if matches else 1
        print('Preserved attempt: '+str(destination))
    else:
        prediction=read(folder,'prediction.json');require_predictions(prediction,['layers'])
        packets=read(folder,'packets.json');manifest=read(folder,'init_manifest.json')
        if (manifest['kind']!='blind' or manifest['packets_sha256']!=base.digest(base.canonical(packets)) or
                packets!=blind_packets(manifest['seed'])):
            raise ValueError('Blind records changed. Keep the task fixed and submit diagnoses; use a new run for a new seed.')
        expected={p['record_id']:diagnose_packet(p) for p in packets}
        if set(prediction['layers']) != set(expected) or any(not isinstance(v,list) for v in prediction['layers'].values()):
            raise ValueError('Predict a list of responsible layers for every record; use ["none"] when clean.')
        matches={k:set(prediction['layers'][k])==set(v) for k,v in expected.items()}
        destination=attempt(folder);save(destination,'submitted_prediction.json',prediction);save(destination,'submitted_packets.json',packets)
        save(destination,'submitted_init_manifest.json',manifest)
        report={'per_record_matches':matches,'all_categories_match':all(matches.values()),'diagnosis':expected,
                'seed':manifest['seed'],'packets_sha256':manifest['packets_sha256'],
                'reasoning_requires_human_review':True,'scope':'Constructed offline fault records, no actual infrastructure fault injection.'}
        save(destination,'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2));print('Preserved attempt: '+str(destination))
        status=0 if all(matches.values()) else 1
    if base.readonly_source_fingerprints()!=before: raise ValueError('Original source fingerprint changed.')
    return status


if __name__=='__main__':
    try: sys.exit(main())
    except (ValueError,KeyError,TypeError,OSError) as error:
        print('Exercise stopped: '+str(error),file=sys.stderr);sys.exit(2)
