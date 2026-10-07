import json
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
def load(n): return json.loads((HERE/n).read_text(encoding='utf-8'))

class Round52PureTests(unittest.TestCase):
    def test_forward_and_loss_preserved(self):
        d=load('FORWARD_AND_GRADIENT_COMPARISON.json')
        self.assertEqual(d['status'],'passed')
        self.assertTrue(d['hook_cleanup_verified'])
        for seed,v in d['per_seed'].items():
            self.assertTrue(v['loss_numeric_equal'])
            self.assertTrue(v['predictions_exactly_equal'])
            self.assertEqual(v['panel_forward_checks'],576)
            self.assertTrue(v['gradient_sum_closed'])

    def test_only_critic_shared_trunk_gradient_is_removed(self):
        d=load('FORWARD_AND_GRADIENT_COMPARISON.json')
        for v in d['per_seed'].values():
            self.assertTrue(v['actor_entropy_gradients_equal'])
            self.assertTrue(v['critic_trunk_isolated'])
            for name in ('task_value','cost_value'):
                self.assertEqual(v['gradient_groups_isolated'][name]['trunk']['l2'],0.0)
            self.assertEqual(v['gradient_groups_old']['task_value']['trunk']['l2']>0,True)
            self.assertEqual(v['gradient_groups_old']['cost_value']['trunk']['l2']>0,True)

    def test_every_real_panel_input_has_old_and_new_direction(self):
        d=load('FORWARD_AND_GRADIENT_COMPARISON.json')
        self.assertEqual(d['input_rows'],576)
        self.assertEqual(d['direction_rows'],1728)
        self.assertEqual(len(d['per_input_directions']),1728)
        for seed in (11,22,33):
            self.assertEqual(sum(r['seed']==seed for r in d['per_input_directions']),576)
        self.assertTrue(all(all(k in r for k in ('unclipped_old','unclipped_isolated','clipped_old','clipped_isolated')) for r in d['per_input_directions']))

    def test_clip_factor_comes_from_frozen_source(self):
        d=load('FORWARD_AND_GRADIENT_COMPARISON.json')
        self.assertEqual(d['per_seed']['11']['clip_norm_source_value'],0.5)
        for v in d['per_seed'].values():
            self.assertAlmostEqual(v['total_old']['l2']*v['old_clip_scale'],0.5,places=5)
            self.assertAlmostEqual(v['total_isolated']['l2']*v['isolated_clip_scale'],0.5,places=5)

    def test_analytic_hook_cleanup_and_zero_operations(self):
        d=load('PURE_TEST_EVIDENCE.json')
        self.assertEqual(d['status'],'passed')
        for k in ('critic_head_detach_preserves_actor_path','exception_hooks_removed','repeated_calls_do_not_accumulate_hooks','wrong_model_role_refusal'):
            self.assertTrue(d[k])
        for k in ('environment_construction','simulation','new_scenes_or_flows','new_evaluation','real_training_updates','synthetic_optimizer_steps','checkpoint_writes'):
            self.assertEqual(d[k],0)

if __name__=='__main__': unittest.main()
