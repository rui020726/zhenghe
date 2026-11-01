# -*- coding: utf-8 -*-
"""ROUND72 validation layer (zero environment, zero forward).

Two independent cross-checks of the tape-derived numbers used by the boundary
audit, plus the calibration-identity check on the focal scene:

1. The free first-report split (sends/skips x early/late) recomputed from the
   persisted tapes is compared with the split published in ROUND68_STATISTICS.json
   and ROUND71_STATISTICS.json.
2. The recorded SEND probability of the focal scene 1700027 is compared across the
   four arms that have a cell there, which is the behavioural form of the
   calibration identity claim (only the two critic heads were replaced).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import R68, R71, SEEDS, MODES, SCENES, FOCAL_SCENE
from r72_boundary_audit import (load_tape, free_decisions, tape_name, ready_step,
                                READY_CUT, HORIZON)


def split_of(tape):
    dec = free_decisions(tape)
    out = {'sends_early': 0, 'skips_early': 0, 'sends_late': 0, 'skips_late': 0}
    for d in dec:
        key = ('sends_' if d['action'] == 1 else 'skips_') + d['region']
        out[key] += 1
    out['total'] = len(dec)
    return out


def main():
    out = {'scope': {'source': 'persisted tapes + published statistics; 0 environment, 0 forward'},
           'first_report_split_recomputed': {}, 'published': {}, 'match': {},
           'focal_probability_identity': {}}

    stats68 = json.loads((R68 / 'raw' / 'ROUND68_STATISTICS.json').read_text(encoding='utf-8'))
    stats71 = json.loads((R71 / 'raw' / 'ROUND71_STATISTICS.json').read_text(encoding='utf-8'))

    for seed in SEEDS:
        for mode in MODES:
            key = '{}/{}'.format(seed, mode)
            a = [load_tape(R68, tape_name('r68', 'trained_alpha', seed, mode, s)) for s in SCENES]
            b = [load_tape(R71, tape_name('r71', 'new_ep128', seed, mode, s)) for s in SCENES]
            rec = {'R68_TRAINED_ALPHA': split_of([r for t in a for r in t]),
                   'NEW_EP128': split_of([r for t in b for r in t])}
            pub68 = (stats68['per_seed_mode'][key].get('first_reports') or {})
            pub71 = (stats71['per_seed_mode'][key].get('first_reports') or {})
            out['first_report_split_recomputed'][key] = rec
            out['published'][key] = {
                'R68_TRAINED_ALPHA': {k: v for k, v in pub68.get('R68_TRAINED_ALPHA', {}).items()
                                      if k != 'unknown_actions'},
                'NEW_EP128': {k: v for k, v in pub71.get('NEW_EP128', {}).items()
                              if k != 'unknown_actions'},
            }
            out['match'][key] = {
                'R68_TRAINED_ALPHA': all(rec['R68_TRAINED_ALPHA'].get(k) == v for k, v in
                                         out['published'][key]['R68_TRAINED_ALPHA'].items()),
                'NEW_EP128': all(rec['NEW_EP128'].get(k) == v for k, v in
                                 out['published'][key]['NEW_EP128'].items()),
            }

    for seed in SEEDS:
        cells = {}
        for tag, arm in (('r68', 'focal_init_alpha'), ('r68', 'focal_trained_alpha'),
                         ('r71', 'focal_new_init'), ('r71', 'focal_new_ep128')):
            name = 'v2_{}eval_{}_s{}_a1_s{}_a1_tape.jsonl'.format(tag, arm, seed, FOCAL_SCENE)
            root = R68 if tag == 'r68' else R71
            cells[arm] = free_decisions(load_tape(root, name))
        a = cells['focal_init_alpha']
        b = cells['focal_new_init']
        c = cells['focal_trained_alpha']
        d = cells['focal_new_ep128']
        out['focal_probability_identity']['s{}'.format(seed)] = {
            'n_free_decisions': len(a),
            'r68_init_vs_r71_new_init_max_abs_dp': max(
                (abs(x['p'] - y['p']) for x, y in zip(a, b)), default=0.0),
            'r68_init_vs_r71_new_init_actions_equal': all(
                x['action'] == y['action'] for x, y in zip(a, b)),
            'r68_init_vs_r71_new_init_probabilities_bit_identical': all(
                x['p'] == y['p'] for x, y in zip(a, b)),
            'r68_trained_vs_r71_new_ep128_max_abs_dp': max(
                (abs(x['p'] - y['p']) for x, y in zip(c, d)), default=0.0),
            'r68_trained_vs_r71_new_ep128_argmax_flips': sum(
                1 for x, y in zip(c, d) if (x['p'] >= 0.5) != (y['p'] >= 0.5)),
            'r68_trained_vs_r71_new_ep128_action_flips': sum(
                1 for x, y in zip(c, d) if x['action'] != y['action']),
            'early_decisions_max_abs_dp': max(
                (abs(x['p'] - y['p']) for x, y in zip(c, d) if x['region'] == 'early'), default=0.0),
            'key_target_decision': [
                {'step': x['step'], 'target': x['target'], 'ready': x['ready'], 'region': x['region'],
                 'p_r68_init': x['p'], 'p_r68_trained': y['p'], 'p_r71_init': z['p'],
                 'p_r71_ep128': w['p'], 'action': x['action']}
                for x, y, z, w in zip(a, c, b, d) if x['target'].endswith('target-5')],
        }
    return out


if __name__ == '__main__':
    res = main()
    dest = Path(__file__).resolve().parent / 'ROUND72_VALIDATION.json'
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', dest, dest.stat().st_size)
    print('split match:', json.dumps(res['match'], ensure_ascii=False))
    print('focal identity:', json.dumps(res['focal_probability_identity'], ensure_ascii=False)[:2500])