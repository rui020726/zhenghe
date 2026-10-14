# -*- coding: utf-8 -*-
"""ROUND72 layer-1 training-signal audit (zero environment, zero forward).

Reads the persisted per-batch training records of ROUND68 and ROUND71 (the same
96 batches per round: 3 seeds x 32 batches) and compares the passive diagnostics
the trainer already wrote: free-row counts by region, surrogate/value losses,
entropy, approx KL, clip fraction, pre-clip per-module gradient L1, the early
gate (prior) loss and the dual lambda update.  Nothing is recomputed, replayed
or re-simulated.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import R68, R71, SEEDS


def load_batches(root, tag):
    out = {}
    ck = root / 'raw' / 'checkpoints'
    for seed in SEEDS:
        rows = []
        for b in range(32):
            name = 'r40_batch_s{}_b{}_attempt01.json'.format(seed, b)
            p = ck / name
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding='utf-8'))
            u = d.get('update') or {}
            gd = u.get('gate_diag') or {}
            rows.append({
                'batch': d.get('batch'), 'seed': d.get('seed'),
                'episodes': d.get('episodes'),
                'lambda_at_rollout': d.get('lambda_at_rollout'),
                'samples': u.get('samples'), 'minibatches': u.get('minibatches'),
                'optimizer_steps': u.get('optimizer_steps'),
                'policy_surrogate': u.get('policy_surrogate'),
                'value_loss_task': u.get('value_loss_task'),
                'value_loss_cost': u.get('value_loss_cost'),
                'entropy': u.get('entropy'), 'approx_kl': u.get('approx_kl'),
                'clip_fraction': u.get('clip_fraction'),
                'gradient_norm_mean': u.get('gradient_norm_mean'),
                'gate_loss': u.get('gate_loss'), 'gate_rows': u.get('gate_rows'),
                'early_free': u.get('early_free'), 'late_free': u.get('late_free'),
                'late_head_l1': (u.get('module_gradient_l1_mean') or {}).get('late_head'),
                'task_value_head_l1': (u.get('module_gradient_l1_mean') or {}).get('task_value_head'),
                'cost_head_l1': (u.get('module_gradient_l1_mean') or {}).get('cost_head'),
                'trunk_l1': (u.get('module_gradient_l1_mean') or {}).get('trunk'),
                'early_head_l1': (u.get('module_gradient_l1_mean') or {}).get('early_head'),
                'gate_diag_update_index': gd.get('update_index'),
                'gate_diag_early_free_rows': gd.get('early_free_rows'),
                'gate_diag_late_free_rows': gd.get('late_free_rows'),
                'late_actor_grad_l1': gd.get('late_actor_grad_l1'),
                'critic_grad_l1': gd.get('critic_grad_l1'),
                'prior_trainable_grad_l1': gd.get('prior_trainable_grad_l1'),
                'prior_late_actor_grad_l1': gd.get('prior_late_actor_grad_l1'),
                'base_trainable_grad_l1': gd.get('base_trainable_grad_l1'),
                'likelihood_max_abs_gap': (u.get('likelihood_alignment') or {}).get('max_abs_gap'),
                'dual_lambda_before': (d.get('dual') or {}).get('lambda_before'),
                'dual_lambda_next': (d.get('dual') or {}).get('lambda_next'),
                'dual_g_hat': (d.get('dual') or {}).get('g_hat'),
                'constraint_g_hat': (d.get('constraint') or {}).get('g_hat'),
                'parameters_before_sha': d.get('parameters_before_sha'),
                'parameters_after_sha': d.get('parameters_after_sha'),
                'structure': d.get('structure'),
            })
        out['s{}'.format(seed)] = rows
    return out


def agg(rows, key):
    xs = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not xs:
        return None
    return {'n': len(xs), 'mean': statistics.fmean(xs), 'min': min(xs), 'max': max(xs),
            'sum': sum(xs)}


def main():
    out = {'scope': {'source': 'persisted per-batch training records; 0 environment, 0 forward, 0 RNG',
                     'batches_per_round': 96, 'rounds': ['round68', 'round71']},
           'rounds': {}}
    for tag, root in (('round68', R68), ('round71', R71)):
        per_seed = load_batches(root, tag)
        allrows = [r for s in per_seed.values() for r in s]
        summary = {}
        for key in ('early_free', 'late_free', 'policy_surrogate', 'value_loss_task',
                    'value_loss_cost', 'entropy', 'approx_kl', 'clip_fraction',
                    'gradient_norm_mean', 'gate_rows', 'late_head_l1',
                    'task_value_head_l1', 'cost_head_l1', 'optimizer_steps',
                    'late_actor_grad_l1', 'critic_grad_l1', 'prior_trainable_grad_l1',
                    'likelihood_max_abs_gap'):
            summary[key] = agg(allrows, key)
        # late actor gradient share of the trainable gradient, where gate_diag exists
        shares = []
        for r in allrows:
            la, cb = r.get('late_actor_grad_l1'), r.get('critic_grad_l1')
            if isinstance(la, (int, float)) and isinstance(cb, (int, float)) and cb > 0:
                shares.append(la / cb)
        summary['late_actor_share_of_critic_grad_l1'] = (
            {'n': len(shares), 'mean': statistics.fmean(shares), 'min': min(shares), 'max': max(shares)}
            if shares else None)
        summary['gate_diag_updates'] = sorted({r['gate_diag_update_index'] for r in allrows
                                               if r['gate_diag_update_index'] is not None})
        summary['batches_with_gate_diag'] = sum(1 for r in allrows if r['gate_diag_update_index'] is not None)
        summary['trunk_l1_all_null'] = all(r['trunk_l1'] is None for r in allrows)
        summary['early_head_l1_all_null'] = all(r['early_head_l1'] is None for r in allrows)
        summary['dual_lambda_values'] = sorted({r['dual_lambda_next'] for r in allrows
                                                if r['dual_lambda_next'] is not None})
        summary['dual_g_hat_values'] = sorted({r['dual_g_hat'] for r in allrows
                                               if r['dual_g_hat'] is not None})
        summary['constraint_g_hat_values'] = sorted({r['constraint_g_hat'] for r in allrows
                                                     if r['constraint_g_hat'] is not None})
        summary['n_batches'] = len(allrows)
        summary['total_optimizer_steps'] = sum(r['optimizer_steps'] or 0 for r in allrows)
        out['rounds'][tag] = {'per_seed_counts': {k: len(v) for k, v in per_seed.items()},
                              'summary': summary,
                              'per_seed': {}}
        for s, rows in per_seed.items():
            ss = {}
            for key in ('early_free', 'late_free', 'policy_surrogate', 'entropy', 'approx_kl',
                        'clip_fraction', 'gradient_norm_mean', 'late_head_l1',
                        'task_value_head_l1', 'cost_head_l1'):
                ss[key] = agg(rows, key)
            ss['total_optimizer_steps'] = sum(r['optimizer_steps'] or 0 for r in rows)
            ss['dual_lambda_values'] = sorted({r['dual_lambda_next'] for r in rows
                                               if r['dual_lambda_next'] is not None})
            out['rounds'][tag]['per_seed'][s] = ss
        out['rounds'][tag]['batches'] = allrows
    return out


if __name__ == '__main__':
    out = main()
    dest = Path(__file__).resolve().parent / 'ROUND72_TRAINING_SIGNAL_AUDIT.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', dest, dest.stat().st_size)
    for tag, r in out['rounds'].items():
        print('==', tag, 'batches', r['summary']['n_batches'],
              'opt steps', r['summary']['total_optimizer_steps'])
        for k in ('early_free', 'late_free', 'policy_surrogate', 'entropy', 'approx_kl',
                  'clip_fraction', 'gradient_norm_mean', 'late_head_l1',
                  'task_value_head_l1', 'cost_head_l1', 'late_actor_grad_l1',
                  'critic_grad_l1', 'late_actor_share_of_critic_grad_l1',
                  'prior_trainable_grad_l1', 'gate_diag_updates', 'batches_with_gate_diag',
                  'trunk_l1_all_null', 'early_head_l1_all_null', 'dual_lambda_values',
                  'dual_g_hat_values', 'constraint_g_hat_values', 'likelihood_max_abs_gap'):
            print('   ', k, json.dumps(r['summary'].get(k), ensure_ascii=False))