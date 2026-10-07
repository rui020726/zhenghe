import json
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent

class Round51PureChecks(unittest.TestCase):
    def test_source_and_no_update_ast_segment(self):
        src=json.loads((HERE/'SOURCE_RECOVERY.json').read_text(encoding='utf-8'))
        grad=json.loads((HERE/'GRADIENT_COMPONENTS.json').read_text(encoding='utf-8'))
        self.assertEqual(src['status'],'exact_physical_source_found_and_verified')
        self.assertTrue(all(v['full_batch_is_not_replayed_minibatch'] for v in grad['per_seed'].values()))
        self.assertEqual(grad['operations']['optimizer_step'],0)
        self.assertEqual(grad['operations']['backward'],0)

    def test_seed_sample_counts_and_gradient_closure(self):
        d=json.loads((HERE/'GRADIENT_COMPONENTS.json').read_text(encoding='utf-8'))
        roles=json.loads((HERE.parent/'round48_policy_drift_probe'/'MODEL_ROLE_IDENTITY.json').read_text(encoding='utf-8'))
        self.assertEqual(set(d['per_seed']),{'11','22','33'})
        for v in d['per_seed'].values():
            self.assertEqual(v['samples'],265)
            self.assertLess(v['gradient_closure']['max_abs'],2e-6)
            seed=next(k for k,x in d['per_seed'].items() if x is v)
            self.assertEqual(v['ep128_identity_verified']['parameter_sha256'],roles['roles']['ep128_final'][seed]['parameter_sha256_recomputed'])

    def test_panel_and_projection_count(self):
        p=json.loads((HERE.parent/'round49_bc_input_policy_closure'/'INPUT_PANEL_MANIFEST.json').read_text(encoding='utf-8'))
        d=json.loads((HERE/'LOGIT_DIRECTIONAL_SENSITIVITY.json').read_text(encoding='utf-8'))
        self.assertEqual(len(p['records']),576)
        self.assertEqual(len(d['records']),3*5*576)
        self.assertTrue(all(len(x['features'])==19 for x in p['records']))

    def test_critic_shared_trunk_path_closed_form(self):
        # Scalar analogue: critic contributes no actor-head gradient, yet moves shared w
        # and thus changes z=v*w*x. This is a pure mathematical guard, not a training update.
        w=torch.tensor(2.0,requires_grad=True); v=torch.tensor(4.0,requires_grad=True)
        u=torch.tensor(5.0,requires_grad=True); x=torch.tensor(3.0); target=torch.tensor(0.0)
        h=w*x; z=v*h; y=u*h; loss=0.5*(y-target)**2
        gw,gv,gu=torch.autograd.grad(loss,(w,v,u),allow_unused=True)
        self.assertIsNone(gv)
        self.assertNotEqual(float(gw),0.0)
        dz_dw=float(v.detach()*x)
        self.assertNotEqual(-float(gw)*dz_dw,0.0)

if __name__=='__main__': unittest.main()
