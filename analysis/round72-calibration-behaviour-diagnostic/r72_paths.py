# -*- coding: utf-8 -*-
"""Resolve ROUND72 input paths without embedding non-ASCII literals."""
from pathlib import Path


def repo_root() -> Path:
    for child in Path('D:/code').iterdir():
        try:
            if child.is_dir() and (child / 'zhenghe-round42').is_dir():
                return child / 'zhenghe-round42'
        except OSError:
            continue
    raise SystemExit('zhenghe-round42 not found under D:/code')


REPO = repo_root()
RUNS = REPO / 'runs' / 'comm_bc_initialized_constrained_ppo_20261007_r2'
R68 = RUNS / 'round68_late_initial_confidence_ablation'
R71 = RUNS / 'round71_critic_initialization_calibration'
R66 = RUNS / 'round66_bc_early_path_preserving_late_adaptation'
R67 = R66 / 'round67_late_adaptation_diagnosis'
DELIV = REPO / 'analysis' / 'round72-calibration-behaviour-diagnostic'

SEEDS = (11, 22, 33)
MODES = ('greedy', 'sampled')
SCENES = tuple(range(1800000, 1800032))
FOCAL_SCENE = 1700027

if __name__ == '__main__':
    for name in ('REPO', 'R68', 'R71', 'R66', 'R67', 'DELIV'):
        print(name, '=', locals()[name], locals()[name].is_dir())