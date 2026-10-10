"""Source-pinned pool/schema and batch preparation. Does not install, infer or train."""
import argparse
import ast
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import platform
import re
import shutil
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('fixed_demo',HERE.parent/'maximum-subarray/run_demo.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
TRAIN_IDS=('climbing-stairs','house-robber','single-number','maximum-product-subarray')
DEV_IDS=('valid-parentheses','longest-substring-without-repeating-characters')


def validate_documented_recipe(text=None):
    """Run the original pure guard on documented scalar overrides, without importing trainer."""
    text=(HERE/'README.md').read_text(encoding='utf-8') if text is None else text
    command=text.split('python -m verl.trainer.main_ppo',1)[1].split('```',1)[0]
    flat={key:value.strip('\"\x27') for key,value in re.findall(r'([A-Za-z_][\w.]*)=([^\s\\]+)',command)}
    def number(key):return int(flat[key]) if key in flat else None
    class Config(dict):
        def __getattr__(self,name):return self[name]
    actor=Config(use_dynamic_bsz=False,ppo_micro_batch_size=number('actor_rollout_ref.actor.ppo_micro_batch_size'),
                 ppo_micro_batch_size_per_gpu=number('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'),
                 ppo_mini_batch_size=number('actor_rollout_ref.actor.ppo_mini_batch_size'),
                 strategy='fsdp',ulysses_sequence_parallel_size=1)
    ref=Config(log_prob_micro_batch_size=number('actor_rollout_ref.ref.log_prob_micro_batch_size'),
               log_prob_micro_batch_size_per_gpu=number('actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu'),
               ulysses_sequence_parallel_size=1)
    rollout=Config(n=number('actor_rollout_ref.rollout.n'),log_prob_micro_batch_size=number('actor_rollout_ref.rollout.log_prob_micro_batch_size'),
                   log_prob_micro_batch_size_per_gpu=number('actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu'))
    cfg=Config(trainer=Config(n_gpus_per_node=number('trainer.n_gpus_per_node'),nnodes=number('trainer.nnodes')),
               data=Config(train_batch_size=number('data.train_batch_size'),val_batch_size=None),
               actor_rollout_ref=Config(actor=actor,ref=ref,rollout=rollout,model=Config(use_remove_padding=True)),
               reward_model=Config(enable=False,use_dynamic_bsz=False))
    source=REPO/'verl/trainer/ppo/ray_trainer.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_validate_config')
    namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
    fake=type('GuardInput',(),{'config':cfg,'use_critic':False})()
    output=io.StringIO()
    with contextlib.redirect_stdout(output):namespace['_validate_config'](fake)
    batch=batch_contract(cfg.data.train_batch_size,rollout.n,cfg.trainer.n_gpus_per_node*cfg.trainer.nnodes,
                         actor.ppo_mini_batch_size,actor.ppo_micro_batch_size_per_gpu,
                         number('actor_rollout_ref.rollout.tensor_model_parallel_size'),1)
    return {'original_validate_config_passed':True,'source_file_sha256':base.digest(source.read_bytes()),
            'selected_overrides':flat,'batch_contract':batch,
            'scope':'Original pure _validate_config method executed on the documented fixed-role scalar configuration. No Hydra/model/CUDA/worker initialization.'}


def batch_contract(q,g,world,mini,micro,tp,sp):
    if any(type(x) is not int or x<=0 for x in (q,g,world,mini,micro,tp,sp)):
        raise ValueError('All batch and parallelism values must be positive integers.')
    if world%tp or world%sp:raise ValueError('World size must be divisible by TP and SP.')
    if (q*g)%world:raise ValueError('Q*G must be divisible by world size.')
    divisor=world//sp
    if (mini*g)%divisor:raise ValueError('Avoid source floor truncation: M*G must divide the FSDP/SP divisor.')
    local_mini=mini*g//divisor;local_rows=q*g//world
    if sp!=1:raise ValueError('This small recipe uses SP=1; other layouts need their own sharding audit.')
    if q%mini or local_mini>local_rows or local_mini%micro:
        raise ValueError('Use complete mini/micro batches: Q%M=0 and normalized mini%micro=0.')
    return {'query_rows':q,'candidates_per_query':g,'global_response_rows':q*g,'world_size':world,
            'local_response_rows':local_rows,'input_mini_batch':mini,'local_mini_batch':local_mini,
            'micro_batch_per_gpu':micro,'gradient_accumulation':local_mini//micro,'rollout_tp':tp,'sp':sp,
            'scope':'Static arithmetic for this source revision, ordinary 1-D FSDP and fixed micro batches.'}


def build_pool():
    base.check_pinned_sources()
    wanted=set(TRAIN_IDS+DEV_IDS);found={};v2_test_ids=set()
    for name in ('LeetCodeDataset-v2-rl-problems.jsonl','LeetCodeDataset-v2-sft-problems.jsonl'):
        path=REPO/'LeetCodeDataset/data'/name
        for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            row=json.loads(line)
            if row['task_id'] in wanted and row['task_id'] not in found:found[row['task_id']]=(row,name,number,line)
    for line in (REPO/'LeetCodeDataset/data/LeetCodeDataset-v2-test-problems.jsonl').read_text(encoding='utf-8').splitlines():
        v2_test_ids.add(json.loads(line)['meta']['question_id'])
    if set(found)!=wanted:raise ValueError('Selected source task missing; review selection.')
    rows={};provenance=[]
    for task_id in TRAIN_IDS+DEV_IDS:
        raw,name,number,line=found[task_id]
        if raw['meta']['question_id'] in v2_test_ids:raise ValueError('Selected task overlaps the source v2 final-test IDs.')
        ast.parse(raw['test'])
        assertions=sum(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(raw['test'])))
        if not assertions:raise ValueError('Empty source tests are refused.')
        statement=raw['meta']['question_title']
        if 'Constraints:' not in statement:raise ValueError('Full statement missing constraints; manual prompt review required.')
        prompt='Please solve this programming task with a self-contained Python code snippet.\n\n'+statement+'\n\nComplete:\n```python\n'+raw['meta']['lang_code']+'\n```'
        split='train' if task_id in TRAIN_IDS else 'dev'
        extra={'split':split,'index':number-1,'prompt':prompt,'dataset':'LeetCodeDataset',
               'reference':raw['completion'],'task_id':task_id,'source_file':'LeetCodeDataset/data/'+name,
               'source_line_1_based':number,'original_corpus_split':raw['meta']['split'],
               'raw_record_sha256':base.digest(line),'test_version':'original-v2-seed-smoke',
               'train_ready':False,'admission':'Seed reuse and schema prepared; per-task answer/prompt/backend review is still required.'}
        row={'data_source':'code','ability':'coding','prompt':[{'role':'system','content':base.extract_system_prompt()},
             {'role':'user','content':prompt}], 'reward_model':{'style':'rule','ground_truth':base.canonical(
             {'functional':raw['test']+'\n\ncheck('+raw['entry_point'].strip()+')'})},'extra_info':extra}
        if raw['completion'] in prompt or raw['test'] in prompt:raise ValueError('Reference/hidden test body leaked into actor prompt.')
        rows[task_id]=row;provenance.append({'task_id':task_id,'file':name,'line':number,'seed_assertions':assertions,'split':split})
    manifest={'source_revision':base.CODE_REV,'dataset_revision':base.DATA_REV,'provenance':provenance,
              'exact_task_overlap':[],'diagnostic_ids_excluded':['maximum-subarray','two-sum'],
              'status':'prepared_not_admitted','uses_public_examples':True,
              'prompt_adaptation':'Use full original statement, including public examples and constraints, plus source signature; not original coder1 meta.query-only preprocessing.',
              'remaining_gates':['Per-task expected/contract and full-statement notation review',
                                 'Semantic-family and experiment split review; distinct IDs alone do not prove independence',
                                 'Target Linux backend isolation, completion, timeout and cleanup checks',
                                 'Runtime dependency/model/config check and baseline before update'],
              'scope':'Tiny train/dev smoke selection. No final blind benchmark or effect claim; no model execution.'}
    return [rows[x] for x in TRAIN_IDS],[rows[x] for x in DEV_IDS],manifest


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='prepare-01');parser.add_argument('--export-parquet',action='store_true')
    parser.add_argument('--world-size',type=int,default=1);parser.add_argument('--q',type=int,default=4)
    parser.add_argument('--g',type=int,default=4);parser.add_argument('--mini',type=int,default=4)
    parser.add_argument('--micro',type=int,default=2);parser.add_argument('--tp',type=int,default=1)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.run):raise ValueError('Run must be a simple name.')
    parent=(HERE/'runs').resolve();folder=(parent/args.run).resolve()
    if not parent.is_relative_to(HERE) or not folder.is_relative_to(parent):raise ValueError('Output path escapes training assets.')
    if folder.exists():raise ValueError('Run already exists; use a new name to preserve prior evidence.')
    before=base.readonly_source_fingerprints();train,dev,manifest=build_pool()
    manifest['batch_contract']=batch_contract(args.q,args.g,args.world_size,args.mini,args.micro,args.tp,1)
    manifest['documented_single_rank_recipe_check']=validate_documented_recipe()
    manifest['current_host']={'platform':platform.system(),'python':platform.python_version(),
                              'nvidia_smi_command_present':shutil.which('nvidia-smi') is not None,
                              'packages':{name:importlib.util.find_spec(name) is not None for name in ('torch','pyarrow','datasets','numpy')}}
    folder.mkdir(parents=True)
    for split,rows in (('train',train),('dev',dev)):
        path=folder/(split+'.jsonl');path.write_text('\n'.join(base.canonical(r) for r in rows)+'\n',encoding='utf-8')
        assert [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]==rows
    manifest['parquet_exported']=False
    manifest_path=folder/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.export_parquet:
        try:import pyarrow as pa;import pyarrow.parquet as pq
        except ImportError:
            manifest['parquet_export_error']='pyarrow absent; no Parquet was written'
            manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            raise ValueError('pyarrow absent. JSONL and a failed-export manifest were retained; install in the target training environment, then use a new run.')
        for split,rows in (('train',train),('dev',dev)):
            table=pa.Table.from_pylist(rows);pq.write_table(table,folder/(split+'.parquet'))
            assert pq.read_table(folder/(split+'.parquet')).to_pylist()==rows
        manifest['parquet_exported']=True
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if base.readonly_source_fingerprints()!=before:raise ValueError('Original source changed.')
    print(json.dumps(manifest,ensure_ascii=False,indent=2));print('Prepared: '+str(folder))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as error:print('Preparation stopped: '+str(error),file=sys.stderr);sys.exit(2)
