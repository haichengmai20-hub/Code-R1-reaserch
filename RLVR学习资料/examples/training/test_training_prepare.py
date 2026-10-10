import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent

def module():
    path=HERE/'prepare_training.py'
    if not path.exists():raise AssertionError('Training preparation entry is absent')
    spec=importlib.util.spec_from_file_location('training_prepare',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class TrainingContract(unittest.TestCase):
    def test_documented_recipe_passes_original_config_guard(self):
        m=module()
        self.assertTrue(hasattr(m,'validate_documented_recipe'),'Original source config check is absent')
        result=m.validate_documented_recipe()
        self.assertTrue(result['original_validate_config_passed'])

    def test_documented_recipe_cannot_set_old_and_new_micro_together(self):
        m=module();text=(HERE/'README.md').read_text(encoding='utf-8')
        modified=text.replace('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2',
                              'actor_rollout_ref.actor.ppo_micro_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2')
        try:
            with self.assertRaises(ValueError):m.validate_documented_recipe(modified)
        except TypeError:self.fail('Recipe variant check is not available yet')

    def test_worker_normalization_matches_concrete_configuration(self):
        m=module();r=m.batch_contract(4,4,2,4,2,1,1)
        self.assertEqual(r['global_response_rows'],16)
        self.assertEqual(r['local_response_rows'],8)
        self.assertEqual(r['local_mini_batch'],8)
        self.assertEqual(r['gradient_accumulation'],4)

    def test_single_rank_with_tp_two_is_rejected(self):
        m=module()
        with self.assertRaises(ValueError):m.batch_contract(4,4,1,4,2,2,1)

    def test_incomplete_micro_batch_is_rejected(self):
        m=module()
        with self.assertRaises(ValueError):m.batch_contract(4,4,2,4,3,1,1)

    def test_pool_uses_distinct_source_tasks_and_no_teaching_diagnostic(self):
        m=module();train,dev,manifest=m.build_pool()
        tid={r['extra_info']['task_id'] for r in train};did={r['extra_info']['task_id'] for r in dev}
        self.assertFalse(tid&did)
        self.assertFalse((tid|did)&{'maximum-subarray','two-sum'})
        self.assertEqual((len(train),len(dev)),(4,2))
        for row in train+dev:
            self.assertFalse(row['extra_info']['train_ready'])
            self.assertIn('prompt',row['extra_info'])
            self.assertIn('index',row['extra_info'])

if __name__=='__main__':unittest.main()
