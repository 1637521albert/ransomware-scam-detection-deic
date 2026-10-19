# Configuration of the expansion experiments reported in Table 2 of the paper,
# used by expansion_experiments.py

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

# Expansion configuration (top panel): the variants are compared with the
# depth and width fixed to the values selected in the paper
CONFIGURATION_VARIANTS = [
    "all over tx",
    "fw bw tx whole",
    "fw bw addr dedicated",
    "fw bw addr whole same",
]
CONFIGURATION_HOPS = 2
CONFIGURATION_LIMIT = 5

# Depth (middle panel) and width (bottom panel) use the selected configuration
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

# Independent of the mode and of the configuration of the pipeline
OUTPUTS_DIR = config.OUTPUT_DIR / "expansion_experiments"
SEEDS_FILE = OUTPUTS_DIR / "seeds.csv"
RUNS_DIR = OUTPUTS_DIR / "runs"
TABLES_DIR = OUTPUTS_DIR / "tables"

def run_name(variant, hops, limit):
    return f"{variant} {hops} hops {limit} {LIMIT_MODE} limit"
