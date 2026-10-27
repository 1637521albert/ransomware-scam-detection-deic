# Configuration of the expansion experiments, used by expansion_experiments.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

SEED = config.DATA_SEED

# Seed addresses: PER_CLASS illicit BitcoinHeist addresses found in the chain, sampled
# at random, and as many licit ones (3,000 in total), selected by
# create_splits.select_seeds. Every run expands all of them into a single graph.
PER_CLASS = 1500

LIMIT_MODE = "random node"

# Expansion configuration: the variants are compared with the depth and width
# fixed to the selected values
CONFIGURATION_VARIANTS = [
    "all over tx",
    "fw bw tx whole",
    "fw bw addr dedicated",
    "fw bw addr whole same",
]
CONFIGURATION_HOPS = 2
CONFIGURATION_LIMIT = 5

# Depth and width use the selected configuration
SELECTED_VARIANT = "fw bw tx whole"
HOPS_VALUES = [1, 2, 3]
HOPS_LIMIT = 5
WIDTH_HOPS = 2
WIDTH_VALUES = [1, 2, 5, 10]

EXPERIMENTS = {
    "configuration": [(variant, CONFIGURATION_HOPS, CONFIGURATION_LIMIT) for variant in CONFIGURATION_VARIANTS],
    "hops": [(SELECTED_VARIANT, hops, HOPS_LIMIT) for hops in HOPS_VALUES],
    "width": [(SELECTED_VARIANT, WIDTH_HOPS, limit) for limit in WIDTH_VALUES],
}

# Runtimes: every run expands its seeds WARMUP_RUNS times without timing them (so the
# blockchain data it reads is in the page cache, whatever ran before it) and then
# TIMED_RUNS times, and its runtime is the median of the timed expansions. All the runs
# of the selected experiments are timed in one session (one execution of the script),
# which only starts on an idle machine: the other processes may use at most
# MAX_OTHER_CORES cores on average over IDLE_CHECK_S seconds, with at least
# MIN_AVAILABLE_MEMORY_PCT of the memory available.
WARMUP_RUNS = 1
TIMED_RUNS = 1
IDLE_CHECK_S = 30
MAX_OTHER_CORES = 1.0
MIN_AVAILABLE_MEMORY_PCT = 75

# Independent of the mode and of the configuration of the pipeline
OUTPUTS_DIR = config.OUTPUT_DIR / "expansion_experiments"
SEEDS_FILE = OUTPUTS_DIR / "seeds.csv"
RUNS_DIR = OUTPUTS_DIR / "runs"
TABLES_DIR = OUTPUTS_DIR / "tables"

def run_name(variant, hops, limit):
    return f"{variant} {hops} hops {limit} {LIMIT_MODE} limit"
