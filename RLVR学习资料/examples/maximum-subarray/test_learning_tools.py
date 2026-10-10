"""Behavior checks for the learning assets, not the original trainer suite."""
import importlib.util
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def tool(name):
    path = HERE / (name + '.py')
    if not path.exists():
        raise AssertionError('Required learning entry is absent: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PracticeBehavior(unittest.TestCase):
    def test_student_entry_exists(self):
        self.assertTrue((HERE / 'practice.py').exists(), 'Student practice entry is absent')

    def test_ninth_valid_pair_is_accepted(self):
        p = tool('practice')
        cases = p.initial_cases('expanded')
        cases.append({'case_id': 'my-ninth', 'input': {'nums': [-2, 4, -1]},
                      'expected': 4, 'origin': 'independent enumeration'})
        report = p.audit_cases(cases)
        self.assertEqual(len(report['accepted']), 9)
        self.assertEqual(len(report['quarantined']), 0)
        self.assertIn("'my-ninth'", p.functional_gt(report['accepted']))

    def test_bad_pair_does_not_remove_correct_same_input(self):
        p = tool('practice')
        cases = [{'case_id': 'good', 'input': {'nums': [1]}, 'expected': 1},
                 {'case_id': 'bad', 'input': {'nums': [1]}, 'expected': 23}]
        report = p.audit_cases(cases)
        self.assertEqual([x['case_id'] for x in report['accepted']], ['good'])
        self.assertEqual([x['case_id'] for x in report['quarantined']], ['bad'])
        self.assertNotIn("'bad'", p.functional_gt(report['accepted']))

    def test_bool_input_and_bool_expected_are_not_integers(self):
        p = tool('practice')
        cases = [{'case_id': 'input-bool', 'input': {'nums': [True]}, 'expected': 1},
                 {'case_id': 'output-bool', 'input': {'nums': [1]}, 'expected': True}]
        report = p.audit_cases(cases)
        self.assertEqual(len(report['accepted']), 0)
        self.assertEqual(len(report['quarantined']), 2)

    def test_large_legal_input_is_pending_not_invalid(self):
        p = tool('practice')
        report = p.audit_cases([{'case_id': 'large', 'input': {'nums': [1] * 65}, 'expected': 65}])
        self.assertEqual(report['pending'][0]['reason'], 'outside_small_oracle_budget')
        self.assertEqual(report['quarantined'], [])

    def test_duplicate_identity_is_rejected_before_execution(self):
        p = tool('practice')
        with self.assertRaises(ValueError):
            p.audit_cases([{'case_id': 'x', 'input': {'nums': [1]}, 'expected': 1}] * 2)

    def test_scorer_changes_only_when_negative_test_added(self):
        p = tool('practice')
        seeds = p.audit_cases(p.initial_cases('seed'))['accepted']
        before = p.score_cases(seeds)
        after = p.score_cases(seeds + p.audit_cases([
            {'case_id': 'my-negative', 'input': {'nums': [-4, -2]}, 'expected': -2}
        ])['accepted'])
        by_id_before = {x['candidate_id']: x for x in before}
        by_id_after = {x['candidate_id']: x for x in after}
        self.assertEqual(by_id_before['wrong_empty']['reward'], 1.1)
        self.assertEqual(by_id_after['wrong_empty']['reward'], 0.1)
        self.assertEqual(by_id_after['reference']['reward'], 1.1)
        self.assertEqual(by_id_after['early_exit']['reward'], 1.1)
        self.assertFalse(by_id_after['early_exit']['correctness_pass'])

    def test_two_sum_order_is_allowed_but_reused_index_is_not(self):
        p = tool('practice')
        self.assertTrue(p.two_sum_valid([2, 7, 11, 15], 9, [1, 0]))
        self.assertFalse(p.two_sum_valid([3, 3], 6, [0, 0]))
        self.assertFalse(p.two_sum_valid([3, 3], 6, [False, 1]))
        self.assertFalse(p.two_sum_valid([3, 3], 6, [-1, 0]))

    def test_all_correct_fault_packet_is_not_forced_to_have_error(self):
        p = tool('practice')
        packet = {'record_id': 'clean', 'input': {'nums': [-2, 4, -1]},
                  'expected': 4, 'gt_expected': 4, 'entry_point': 'Solution().maxSubArray',
                  'scorer_reward': 1.1, 'observed_cases': 1, 'required_cases': 1,
                  'claims_correctness': True, 'actual_values': [4],
                  'n': 4, 'c': 2, 'k': 2, 'reported_pass_at_k': 5/6}
        self.assertEqual(p.diagnose_packet(packet), ['none'])
        packet['expected'] = 40
        self.assertIn('Tests', p.diagnose_packet(packet))
        self.assertIn('GT', p.diagnose_packet(packet))

    def test_full_count_with_wrong_actual_is_not_correctness(self):
        p = tool('practice')
        packet = p.blind_packets(53)[0]
        packet.update(input={'nums': [1]}, expected=1, gt_expected=1,
                      entry_point='Solution().maxSubArray', scorer_reward=1.1,
                      observed_cases=1, required_cases=1, claims_correctness=True,
                      actual_values=[2], n=4, c=2, k=2, reported_pass_at_k=5/6)
        self.assertEqual(p.diagnose_packet(packet), ['Verifier'])

    def test_output_run_name_cannot_escape_exercise_directory(self):
        p = tool('practice')
        with self.assertRaises(ValueError):
            p.run_directory('../outputs')

    def test_transfer_rejects_equal_value_with_changed_numeric_type(self):
        p = tool('practice')
        self.assertTrue(hasattr(p, 'assert_transfer_material'), 'Typed material integrity gate is missing')
        original=p.transfer_material()
        import copy
        changed=copy.deepcopy(original)
        changed['tests'][0]['input']['nums'][0]=float(changed['tests'][0]['input']['nums'][0])
        with self.assertRaises(ValueError):p.assert_transfer_material(changed, original)

    def test_blind_seed_changes_material_not_only_identifiers(self):
        p = tool('practice')
        one={tuple(x['input']['nums']) for x in p.blind_packets(53)}
        two={tuple(x['input']['nums']) for x in p.blind_packets(54)}
        self.assertNotEqual(one,two)

    def test_assembled_transfer_GT_accepts_both_orders_and_rejects_reuse(self):
        p = tool('practice');material=p.transfer_material();results=p.transfer_results(material)
        p.transfer_gt_regression(material,results)
        self.assertEqual(results['A']['semantic_reward'],1.1)
        self.assertEqual(results['B']['semantic_reward'],1.1)
        self.assertEqual(results['B']['legacy_reward'],.1)
        self.assertEqual(results['C']['semantic_reward'],.1)


class GradientBehavior(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        m = tool('mechanism_lab')
        logits = [0.0, math.log(3)]
        loss, gradient, info = m.policy_loss_and_gradient(logits, 1, math.log(.75), .5, .2)
        numeric = m.finite_difference(logits, 1, math.log(.75), .5, .2)
        self.assertAlmostEqual(loss, -.5)
        self.assertAlmostEqual(gradient[0], .125)
        self.assertAlmostEqual(gradient[1], -.125)
        for a, b in zip(gradient, numeric): self.assertAlmostEqual(a, b, places=6)

    def test_positive_signal_increases_selected_probability_before_clip(self):
        m = tool('mechanism_lab')
        logits = [0.0, math.log(3)]
        _, gradient, _ = m.policy_loss_and_gradient(logits, 1, math.log(.75), .5, .2)
        changed = [z-.1*g for z,g in zip(logits, gradient)]
        self.assertGreater(m.softmax(changed)[1], m.softmax(logits)[1])

    def test_clip_saturation_can_zero_policy_gradient(self):
        m = tool('mechanism_lab')
        logits = [0.0, math.log(9)]
        _, gradient, info = m.policy_loss_and_gradient(logits, 1, math.log(.5), 1., .2)
        self.assertTrue(info['clipped_branch'])
        self.assertEqual(gradient, [0., 0.])

    def test_negative_advantage_does_not_share_positive_clip_branch(self):
        m = tool('mechanism_lab')
        logits = [0.0, math.log(9)]
        _, gradient, info = m.policy_loss_and_gradient(logits, 1, math.log(.5), -1., .2)
        self.assertFalse(info['clipped_branch'])
        self.assertGreater(gradient[1], 0.)

    def test_length_weight_and_eos_are_separate_from_padding(self):
        m = tool('mechanism_lab')
        self.assertEqual(m.masked_mean([[1.,1.,0.,0.,0.,0.],[3.]*6],
                                     [[1,1,0,0,0,0],[1]*6]), 2.5)
        self.assertEqual(m.eos_mask([4,2,2,2], 2), [1,1,0,0])
        self.assertEqual(m.eos_mask([4,5,6], 2), [1,1,1])


if __name__ == '__main__':
    unittest.main()
