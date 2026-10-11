# Shared configuration of the k (hops) / w (random node width limit) expansion
# experiments, used by expansion_experiments.py and experiments_metrics.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import build_run_dir, build_run_name

seed = "licit and illicit"

direction = "fw bw"
graph = " tx"
address = " whole"
side_direction = ""
exp_alg = f'{direction}{graph}{address}{side_direction}'

limit_mode = "random node"
space = " "

# Number of illicit seeds per split (the same number of licit seeds is added)
train_samples = 3000
val_samples = 1000
test_samples = 1000

SPLITS = ['train', 'val', 'test']

# (hops, random node width limit)
HOPS_EXPERIMENTS = [(hops, 5) for hops in (1, 2, 3)]
WIDTH_EXPERIMENTS = [(2, limit) for limit in (1, 2, 5, 10)]
# (2 hops, limit 5) belongs to both sweeps, so it is only run once
EXPERIMENTS = list(dict.fromkeys(HOPS_EXPERIMENTS + WIDTH_EXPERIMENTS))

def experiment_run_dir(hops, limit):
    run_name = build_run_name(
        train_samples,
        val_samples,
        test_samples,
        exp_alg,
        hops,
        limit,
        space,
        limit_mode,
    )
    return build_run_dir(__file__, run_name)
