# -*- coding: utf-8 -*-
"""ROUND72 layer-2 boundary audit (zero environment, zero forward).

Reads the persisted pre-action decision-step tapes of the ROUND68 and ROUND71
evaluation cells on the already-used development block 1800000..1800031 and
compares, per free first-report decision, the recorded SEND probability and the
recorded action between model arms.  Nothing is re-simulated, no model is loaded,
no RNG is touched: every number below is a persisted field of a real run.

Recorded fields used (all produced by the real controller at run time):
  features[i]        the pre-action legal 19-D observation of candidate i
  send_probabilities softmax over the two one-shot actions of that candidate
  actions[i]         0 = SKIP, 1 = SEND
Greedy decoding is exactly `SEND iff p >= 0.5` (r42_one_shot.py), so the greedy
decision boundary is p = 0.5 and `argmax_flips` below is the decision-level
boundary-crossing count.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r72_paths import R68, R71, SEEDS, MODES, SCENES, FOCAL_SCENE

HORIZON = 1000.0
READY_CUT = 147.0


def ready_step(feature):
    return int(round(float(feature[18]) * HORIZON) - round(float(feature[0])))


def tape_name(round_tag, arm, seed, mode, scene):
    return 'v2_{}eval_model_{}_s{}_{}_a1_s{}_a1_tape.jsonl'.format(round_tag, arm, seed, mode, scene)


def load_tape(root, name):
    path = root / 'raw' / 'logs' / name
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def free_decisions(tape):
    out = []
    for rec in tape:
        free = rec.get('free') or []
        if not free:
            continue
        targets = rec['targets']
        feats = rec['features']
        probs = rec.get('send_probabilities') or {}
        audit = rec.get('candidate_audit') or {}
        for t in free:
            i = targets.index(t)
            f = feats[i]
            r = ready_step(f)
            out.append({
                'step': int(rec['step']), 'target': t, 'ready': r,
                'region': 'early' if r <= READY_CUT else 'late',
                'p': float(probs[t]), 'action': int(rec['actions'][t]),
                'charge': (audit.get(t) or {}).get('attempt_charge_bytes'),
            })
    return out


def trajectory_signature(tape):
    return ([int(r['step']) for r in tape],
            json.dumps([r['features'] for r in tape], sort_keys=True),
            json.dumps([r['actions'] for r in tape], sort_keys=True))


def compare_cell(ta, tb):
    sa, fa, aa = trajectory_signature(ta)
    sb, fb, ab = trajectory_signature(tb)
    ident = {'steps_equal': sa == sb, 'features_equal': fa == fb, 'actions_equal': aa == ab}
    da = {(d['step'], d['target']): d for d in free_decisions(ta)}
    db = {(d['step'], d['target']): d for d in free_decisions(tb)}
    rows = []
    for k in da:
        if k not in db:
            continue
        x, y = da[k], db[k]
        rows.append({'step': k[0], 'target': k[1], 'ready': x['ready'], 'region': x['region'],
                     'p_a': x['p'], 'p_b': y['p'], 'act_a': x['action'], 'act_b': y['action'],
                     'dp': y['p'] - x['p'], 'charge': x['charge']})
    return ident, rows, len(set(da) - set(db)), len(set(db) - set(da))


def margin_block(rows):
    """Boundary geometry of the matched free decisions."""
    late = [r for r in rows if r['region'] == 'late']
    early = [r for r in rows if r['region'] == 'early']
    toward, away, neutral = [], [], []
    for r in late:
        m = r['p_a'] - 0.5
        if r['dp'] == 0.0:
            neutral.append(r)
        elif (r['dp'] > 0) == (m > 0):
            away.append(r)
        else:
            toward.append(r)
    enc = [abs(r['dp']) / abs(r['p_a'] - 0.5) for r in toward if r['p_a'] != 0.5]
    return {
        'early_max_abs_dp': max((abs(r['dp']) for r in early), default=0.0),
        'early_n_dp_nonzero': sum(1 for r in early if r['dp'] != 0.0),
        'late_n': len(late),
        'late_n_dp_nonzero': sum(1 for r in late if r['dp'] != 0.0),
        'late_max_abs_dp': max((abs(r['dp']) for r in late), default=0.0),
        'late_min_abs_margin_a': min((abs(r['p_a'] - 0.5) for r in late), default=None),
        'late_min_abs_margin_b': min((abs(r['p_b'] - 0.5) for r in late), default=None),
        'late_p_a_min': min((r['p_a'] for r in late), default=None),
        'late_p_a_max': max((r['p_a'] for r in late), default=None),
        'late_moved_toward_boundary': len(toward),
        'late_moved_away_from_boundary': len(away),
        'late_dp_exactly_zero': len(neutral),
        'late_max_encroachment_ratio': max(enc, default=None),
        'argmax_flips': sum(1 for r in rows if (r['p_a'] >= 0.5) != (r['p_b'] >= 0.5)),
    }


def summarise(rows):
    early = [r for r in rows if r['region'] == 'early']
    late = [r for r in rows if r['region'] == 'late']
    return {
        'n_free': len(rows),
        'early': {'n': len(early),
                  'sends': sum(1 for r in early if r['act_a'] == 1),
                  'skips': sum(1 for r in early if r['act_a'] == 0)},
        'late': {'n': len(late),
                 'sends': sum(1 for r in late if r['act_a'] == 1),
                 'skips': sum(1 for r in late if r['act_a'] == 0)},
        'action_flips': sum(1 for r in rows if r['act_a'] != r['act_b']),
        'margin': margin_block(rows),
    }


def main():
    comparisons = {
        'PRIMARY_R68_TRAINED_VS_R71_NEW_EP128': ('r68', 'trained_alpha', 'r71', 'new_ep128'),
        'SECONDARY_R68_INIT_VS_R68_TRAINED': ('r68', 'init_alpha', 'r68', 'trained_alpha'),
        'SECONDARY_R68_INIT_VS_R71_NEW_EP128': ('r68', 'init_alpha', 'r71', 'new_ep128'),
    }
    roots = {'r68': R68, 'r71': R71}
    result = {'scope': {
        'seeds': list(SEEDS), 'modes': list(MODES),
        'scenes': [SCENES[0], SCENES[-1]], 'n_scenes': len(SCENES),
        'block': '1800000..1800031 (already-used development block, not a confirmation set)',
        'source': 'persisted decision-step tapes of real runs; 0 environment constructions, '
                  '0 model loads, 0 forward passes, 0 RNG use',
        'greedy_rule': 'SEND iff recorded p(SEND) >= 0.5 (r42_one_shot.py:223-225)',
        'routing': 'ready_step = round(f[18]*1000) - round(f[0]); early iff ready_step <= 147',
    }, 'comparisons': {}, 'focal': {}, 'decisions': {}, 'sampled_flips': {}}

    for cname, (ta, arma, tb, armb) in comparisons.items():
        per_cell, all_rows, cells_missing = {}, [], []
        for seed in SEEDS:
            for mode in MODES:
                rows = []
                ident = {'steps_equal': 0, 'features_equal': 0, 'actions_equal': 0}
                ncell, orphan = 0, 0
                for scene in SCENES:
                    na, nb = tape_name(ta, arma, seed, mode, scene), tape_name(tb, armb, seed, mode, scene)
                    pa, pb = roots[ta] / 'raw' / 'logs' / na, roots[tb] / 'raw' / 'logs' / nb
                    if not (pa.exists() and pb.exists()):
                        cells_missing.append([na, nb])
                        continue
                    ci, crows, oa, ob = compare_cell(load_tape(roots[ta], na), load_tape(roots[tb], nb))
                    ident['steps_equal'] += int(ci['steps_equal'])
                    ident['features_equal'] += int(ci['features_equal'])
                    ident['actions_equal'] += int(ci['actions_equal'])
                    orphan += oa + ob
                    ncell += 1
                    for r in crows:
                        r2 = dict(r)
                        r2.update({'seed': seed, 'mode': mode, 'scene': scene})
                        rows.append(r2)
                per_cell['s{}_{}'.format(seed, mode)] = {
                    'cells': ncell, 'identity': ident,
                    'decision_keys_present_in_only_one_arm': orphan,
                    'summary': summarise(rows)}
                all_rows.extend(rows)
        result['comparisons'][cname] = {
            'arms': [ta + ':' + arma, tb + ':' + armb],
            'cells_missing': cells_missing,
            'per_seed_mode': per_cell,
            'overall': summarise(all_rows)}
        if cname.startswith('PRIMARY'):
            result['decisions'] = all_rows
            flips = [r for r in all_rows if r['act_a'] != r['act_b']]
            result['sampled_flips'] = {
                'n': len(flips),
                'all_in_sampled_mode': all(r['mode'] == 'sampled' for r in flips),
                'detail': [{'seed': r['seed'], 'scene': r['scene'], 'step': r['step'],
                            'target': r['target'], 'region': r['region'], 'ready': r['ready'],
                            'p_a': r['p_a'], 'p_b': r['p_b'], 'dp': r['dp'],
                            'act_a': r['act_a'], 'act_b': r['act_b'],
                            'p_bit_identical': r['p_a'] == r['p_b']} for r in flips],
                'sampled_matched_decisions': sum(1 for r in all_rows if r['mode'] == 'sampled'),
                'sampled_decisions_with_changed_p': sum(1 for r in all_rows if r['mode'] == 'sampled' and r['dp'] != 0.0),
                'sampled_decisions_with_identical_p': sum(1 for r in all_rows if r['mode'] == 'sampled' and r['dp'] == 0.0),
            }

    focal = {}
    focal_arms = (('r68', 'focal_init_alpha'), ('r68', 'focal_trained_alpha'),
                  ('r71', 'focal_new_init'), ('r71', 'focal_new_ep128'))
    for seed in SEEDS:
        cells = {}
        for tag, arm in focal_arms:
            name = 'v2_{}eval_{}_s{}_a1_s{}_a1_tape.jsonl'.format(tag, arm, seed, FOCAL_SCENE)
            p = roots[tag] / 'raw' / 'logs' / name
            if not p.exists():
                cells[tag + ':' + arm] = {'missing': True}
                continue
            cells[tag + ':' + arm] = [
                {'step': d['step'], 'target': d['target'], 'ready': d['ready'], 'region': d['region'],
                 'p': d['p'], 'action': d['action'], 'charge': d['charge']}
                for d in free_decisions(load_tape(roots[tag], name))]
        focal['s{}'.format(seed)] = cells
    result['focal'] = focal
    return result


if __name__ == '__main__':
    out = main()
    dest = Path(__file__).resolve().parent / 'ROUND72_BOUNDARY_AUDIT.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', dest, dest.stat().st_size)
    for cname, c in out['comparisons'].items():
        o = c['overall']
        print('==', cname)
        print('   n_free', o['n_free'], 'action_flips', o['action_flips'])
        print('   margin', json.dumps(o['margin'], ensure_ascii=False))
        for k, v in c['per_seed_mode'].items():
            m = v['summary']['margin']
            print('   ', k, 'cells', v['cells'], v['identity'],
                  'flips', m['argmax_flips'], 'actflips', v['summary']['action_flips'],
                  'late_n', m['late_n'], 'late_dp!=0', m['late_n_dp_nonzero'],
                  'late_min_margin_a', m['late_min_abs_margin_a'])
    print('sampled_flips', json.dumps(out['sampled_flips'], ensure_ascii=False)[:1500])