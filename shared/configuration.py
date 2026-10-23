# Parameters of the experiments that the user can choose, and the configuration files
# that store them.
#
# config.py holds the parameters of the original configuration. A custom
# configuration replaces some of them: run_pipeline.py asks them to the user (ask),
# saves them in a JSON file (save) and gives the file to every step through
# RSD_CONFIG_FILE, so that config.py applies it (apply). The file has one entry per
# parameter of config.py, e.g. {"DATA_SEED": 42, "GNN_TRAINING": {"patience": 10, ...}}.
#
# Only the parameters that change the data or the results are configurable; the
# resources (workers, memory) stay in config.py. The identifier of a configuration is
# a hash of all its parameters, and names its output directory.

import copy
import hashlib
import json
import sys

# --------------------------------------------------------------------------
# Expansions
# --------------------------------------------------------------------------

# Expansion variants: name -> (direction, graph, address, side_direction), as in config.py.
# Both methodologies implement every variant (shared/expansion.py).
MULTI_INSTANCE_VARIANTS = {
    "fw bw tx whole": ("fw bw", " tx", " whole", ""),
    "fw bw tx dedicated": ("fw bw", " tx", " dedicated", ""),
    "all over tx": ("all over", " tx", "", ""),
    "fw bw addr whole none": ("fw bw", " addr", " whole", " none"),
    "fw bw addr whole same": ("fw bw", " addr", " whole", " same"),
    "fw bw addr whole opposite": ("fw bw", " addr", " whole", " opposite"),
    "fw bw addr dedicated": ("fw bw", " addr", " dedicated", ""),
    "all over addr": ("all over", " addr", "", ""),
}
EGO_CENTRIC_VARIANTS = MULTI_INSTANCE_VARIANTS
EXPANSIONS = {"MULTI_INSTANCE_EXPANSION": MULTI_INSTANCE_VARIANTS, "EGO_CENTRIC_EXPANSION": EGO_CENTRIC_VARIANTS}

# Width limit of the expansion: "" (none) is shown as "none"
LIMIT_MODES = ["random node", "random hop", "none"]


def variant_name(expansion, variants):
    key = (expansion["direction"], expansion["graph"], expansion["address"], expansion["side_direction"])
    for name, value in variants.items():
        if value == key:
            return name
    raise ValueError(f"unknown expansion variant {key}; available: {', '.join(variants)}")


def check_expansion(expansion, variants):
    """Validated copy of an expansion dictionary (raises ValueError)."""
    if not isinstance(expansion, dict):
        raise ValueError("an expansion must be a dictionary")
    name = variant_name(expansion, variants)
    limit_mode = expansion.get("limit_mode", "")
    if limit_mode not in ("random node", "random hop", ""):
        raise ValueError(f"limit_mode must be 'random node', 'random hop' or '' (none), not {limit_mode!r}")
    checked = dict(zip(["direction", "graph", "address", "side_direction"], variants[name]))
    checked["limit_mode"] = limit_mode
    checked["limit"] = check_int(expansion.get("limit"), 1) if limit_mode else None
    hop_keys = ["forward_hops", "backward_hops"] if name.startswith("fw bw") else ["hops"]
    for key in hop_keys:
        checked[key] = check_int(expansion.get(key), 1)
    unknown = set(expansion) - set(checked)
    if unknown:
        raise ValueError(f"unknown expansion keys {sorted(unknown)} for the {name} expansion")
    return checked


def describe_expansion(expansion, variants):
    name = variant_name(expansion, variants)
    limit = f"{expansion['limit_mode']} w={expansion['limit']}" if expansion["limit_mode"] else "no width limit"
    if name.startswith("fw bw"):
        hops = f"{expansion['forward_hops']} forward + {expansion['backward_hops']} backward hops"
    else:
        hops = f"k={expansion['hops']} hops"
    return f"{name}, {limit}, {hops}"


def ask_expansion(original, variants):
    names = list(variants)
    name = ask("Expansion variant", variant_name(original, variants), choice_parser(names),
               f"one of: {', '.join(names)}") if len(names) > 1 else names[0]
    expansion = dict(zip(["direction", "graph", "address", "side_direction"], variants[name]))
    limit_mode = ask("Width limit", original["limit_mode"] or "none", choice_parser(LIMIT_MODES),
                     f"one of: {', '.join(LIMIT_MODES)}")
    expansion["limit_mode"] = "" if limit_mode == "none" else limit_mode
    expansion["limit"] = (ask("Width limit w (transactions per node or hop)", original.get("limit") or 5, int_parser(1))
                          if expansion["limit_mode"] else None)
    if name.startswith("fw bw"):
        expansion["forward_hops"] = ask("Forward hops", original.get("forward_hops", original.get("hops", 2)), int_parser(1))
        expansion["backward_hops"] = ask("Backward hops", original.get("backward_hops", original.get("hops", 2)), int_parser(1))
    else:
        expansion["hops"] = ask("Hops k", original.get("hops", original.get("forward_hops", 2)), int_parser(1))
    return expansion

# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------

def check_int(value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{value!r} is not an integer")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{value} is not between {low} and {high}" if high is not None else f"{value} is lower than {low}")
    return value


def check_float(value, low=None, high=None, low_open=False, high_open=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{value!r} is not a number")
    value = float(value)
    if low is not None and (value < low or (low_open and value == low)):
        raise ValueError(f"{value} must be {'greater than' if low_open else 'at least'} {low}")
    if high is not None and (value > high or (high_open and value == high)):
        raise ValueError(f"{value} must be {'lower than' if high_open else 'at most'} {high}")
    return value


def int_parser(low=None, high=None):
    return lambda text: check_int(int(text), low, high)


def float_parser(low=None, high=None, low_open=False, high_open=False):
    return lambda text: check_float(float(text), low, high, low_open, high_open)


def choice_parser(choices):
    lookup = {c.lower(): c for c in choices}
    def parse(text):
        if text.strip().lower() not in lookup:
            raise ValueError(f"{text.strip()!r} is not one of {', '.join(choices)}")
        return lookup[text.strip().lower()]
    return parse


def subset_parser(choices):
    choice = choice_parser(choices)
    def parse(text):
        values = [choice(part) for part in text.split(",") if part.strip()]
        if not values:
            raise ValueError("choose at least one")
        if len(set(values)) != len(values):
            raise ValueError("repeated values")
        return values
    return parse


def int_list_parser(low=1):
    def parse(text):
        values = [check_int(int(part), low) for part in text.split(",") if part.strip()]
        if not values:
            raise ValueError("give at least one number")
        return values
    return parse


def word_or(word, parser):
    """A parser that also accepts a keyword (e.g. 'exact' for None, 'auto')."""
    value = None if word == "exact" else word
    return lambda text: value if text.strip().lower() == word else parser(text)


def show(value):
    if value is None:
        return "exact"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)

# --------------------------------------------------------------------------
# Configurable parameters
# --------------------------------------------------------------------------

# Columns of the ego-centric features (extract_topological_features.py)
EGO_CENTRIC_AVAILABLE_FEATURES = [
    "star_out", "2star_out", "split_2+", "star_in", "2star_in", "merge_2+",
    "in_deg", "out_deg", "peeling_2", "peeling_3",
    "pagerank", "clustering", "betweenness", "eccentricity", "density", "n_nodes", "n_edges",
]
EGO_CENTRIC_MODELS = ["L1 (Lasso)", "ElasticNet", "RandomForest", "XGBoost", "IsolationForest", "SVM", "Naive Bayes", "MLP"]
GNN_ARCHITECTURES = ["GAT", "HGT", "HAN"]


class Param:
    """One parameter: config.py attribute (and key, for the dictionaries), question and parser.
    The parser reads the answer of the user; a value from a file is checked by parsing its text."""

    def __init__(self, attr, key, label, parser, help_text="", as_tuple=False):
        self.attr, self.key, self.label, self.parser = attr, key, label, parser
        self.help_text, self.as_tuple = help_text, as_tuple

    def get(self, params):
        return params[self.attr][self.key] if self.key else params[self.attr]

    def set(self, params, value):
        if self.key:
            params[self.attr][self.key] = value
        else:
            params[self.attr] = value

    def check(self, value):
        return self.parser(show(value) if value is not None else "exact")


# Groups of parameters, by methodology ("shared": every methodology)
GROUPS = [
    ("shared", "Data generation", [
        Param("DATA_SEED", None, "Seed of the data generation (new splits, expansions, extra training seeds)", int_parser(0)),
    ]),
    ("multi-instance", "Methodology I - expansion", "MULTI_INSTANCE_EXPANSION"),
    ("multi-instance", "Methodology I - GNN models and training", [
        Param("GNN_ARCHITECTURES", None, "Architectures", subset_parser(GNN_ARCHITECTURES),
              f"comma-separated, from: {', '.join(GNN_ARCHITECTURES)}"),
        Param("GNN_LAYERS", None, "Numbers of layers", int_list_parser(1), "comma-separated"),
        Param("GNN_TRAINING", "hidden_channels_GAT", "Hidden channels of GAT (a multiple of the heads)", int_parser(1)),
        Param("GNN_TRAINING", "hidden_channels_HGT", "Hidden channels of HGT (a multiple of the heads)", int_parser(1)),
        Param("GNN_TRAINING", "hidden_channels_HAN", "Hidden channels of HAN (a multiple of the heads)", int_parser(1)),
        Param("GNN_TRAINING", "heads", "Attention heads per layer", int_parser(1)),
        Param("GNN_TRAINING", "learning_rate", "Learning rate", float_parser(0, low_open=True)),
        Param("GNN_TRAINING", "dropout", "Dropout", float_parser(0, 1, high_open=True)),
        Param("GNN_TRAINING", "label_smoothing", "Label smoothing of the training loss (0: none)",
              float_parser(0, 1, high_open=True)),
        Param("GNN_TRAINING", "ema_decay", "Decay of the moving average of the weights that is evaluated (0: none)",
              float_parser(0, 1, high_open=True)),
        Param("GNN_TRAINING", "patience", "Early-stopping patience (epochs)", int_parser(1)),
    ]),
    ("ego-centric", "Methodology II - expansion", "EGO_CENTRIC_EXPANSION"),
    ("ego-centric", "Methodology II - topological features", [
        Param("EGO_CENTRIC_FEATURES", "radius", "BFS radius around the seed (address-transaction steps)", int_parser(1)),
        Param("EGO_CENTRIC_FEATURES", "betweenness_k", "Betweenness", word_or("exact", int_parser(1)),
              "'exact', or the number of sampled nodes of the approximation"),
    ]),
    ("ego-centric", "Methodology II - tabular models", [
        Param("EGO_CENTRIC_TRAINING_FEATURES", None, "Training features", subset_parser(EGO_CENTRIC_AVAILABLE_FEATURES),
              f"comma-separated, from: {', '.join(EGO_CENTRIC_AVAILABLE_FEATURES)}"),
    ]),
    ("shared", "Statistical tests", [
        Param("BOOTSTRAP_RESAMPLES", None, "Bootstrap resamples of the confidence intervals", int_parser(100)),
        Param("MIN_SEEDS_FOR_TESTS", None, "Minimum common seeds of a Wilcoxon test", int_parser(2)),
    ]),
]

PARAMS = [p for _, _, params in GROUPS if isinstance(params, list) for p in params]


def jsonable(value):
    return json.loads(json.dumps(value))


def parameters(namespace):
    """The configurable parameters of a config.py namespace, as a JSON-serializable dictionary."""
    params = {}
    for p in PARAMS:
        value = namespace[p.attr][p.key] if p.key else namespace[p.attr]
        if p.key:
            params.setdefault(p.attr, {})[p.key] = value
        else:
            params[p.attr] = value
    for attr in EXPANSIONS:
        params[attr] = dict(namespace[attr])
    return jsonable(params)


def identifier(params):
    """Name of a custom configuration, from a hash of all its parameters."""
    text = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return "custom-" + hashlib.sha256(text.encode()).hexdigest()[:10]


# Parameters of earlier versions, ignored in configuration files (the interpreted
# ego-centric model is now the best one of every run)
OBSOLETE = {"EGO_CENTRIC_BEST_MODEL"}


def check(params):
    """Validated copy of a dictionary of parameters (raises ValueError)."""
    params = {k: v for k, v in params.items() if k not in OBSOLETE}
    checked = copy.deepcopy(params)
    known = {p.attr for p in PARAMS} | set(EXPANSIONS)
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"unknown parameters {unknown}")
    errors = []
    for attr, variants in EXPANSIONS.items():
        if attr in params:
            try:
                checked[attr] = check_expansion(params[attr], variants)
            except ValueError as e:
                errors.append(f"{attr}: {e}")
    for p in PARAMS:
        if p.attr not in params or (p.key and p.key not in params[p.attr]):
            continue
        try:
            p.set(checked, p.check(p.get(params)))
        except (ValueError, TypeError) as e:
            errors.append(f"{p.attr}{'.' + p.key if p.key else ''}: {e}")
    for attr in {p.attr for p in PARAMS if p.key} & set(params):
        keys = {p.key for p in PARAMS if p.attr == attr}
        extra = sorted(set(params[attr]) - keys)
        if extra:
            errors.append(f"{attr}: unknown keys {extra}")
    if errors:
        raise ValueError("Invalid configuration:\n  " + "\n  ".join(errors))
    return checked


def apply(namespace, path):
    """Replaces the parameters of a config.py namespace with those of a configuration file."""
    with open(path, encoding="utf-8") as f:
        params = check(json.load(f))
    for attr, value in params.items():
        if attr in EXPANSIONS:
            namespace[attr] = value
        elif isinstance(value, dict):
            namespace[attr] = dict(namespace[attr], **value)
        else:
            namespace[attr] = value
    for p in PARAMS:
        if p.as_tuple and p.key:
            namespace[p.attr][p.key] = tuple(namespace[p.attr][p.key])


def save(params, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
        f.write("\n")

# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------

def ask(label, default, parser, help_text=""):
    """Asks one value until it is valid; Enter keeps the default."""
    hint = f" ({help_text})" if help_text else ""
    while True:
        answer = input(f"  {label}{hint} [{show(default)}]: ").strip()
        if not answer:
            return parser(show(default))
        try:
            return parser(answer)
        except (ValueError, TypeError) as e:
            print(f"    Invalid value: {e}")


def ask_yes(question, default=True):
    while True:
        answer = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "n", "no"):
            return answer.startswith("y")
        print("    Answer y or n.")


def describe_group(params, group):
    if isinstance(group, str):
        return describe_expansion(params[group], EXPANSIONS[group])
    return "; ".join(f"{p.label.lower()} {show(p.get(params))}" for p in group[:4]) + ("; ..." if len(group) > 4 else "")


def ask_configuration(original, methodologies):
    """Asks the parameters of the selected methodologies (and the shared ones), starting
    from the original ones. Returns the complete dictionary of parameters."""
    if not sys.stdin.isatty():
        raise SystemExit("Choosing the parameters needs an interactive terminal: run with --original "
                         "(the original configuration) or --config <file> (a saved configuration).")
    params = copy.deepcopy(original)
    print("\nConfiguration of the experiments. For every group, keep the original parameters or "
          "choose new ones (Enter keeps the value in brackets).")
    for methodology, title, group in GROUPS:
        if methodology != "shared" and methodology not in methodologies:
            continue
        print(f"\n{title}: {describe_group(original, group)}")
        if ask_yes("  Keep these parameters?"):
            continue
        if isinstance(group, str):
            params[group] = ask_expansion(original[group], EXPANSIONS[group])
        else:
            for p in group:
                p.set(params, ask(p.label, p.get(original), p.parser, p.help_text))
    gnn = params["GNN_TRAINING"]
    for arch in ("GAT", "HGT", "HAN"):
        if gnn[f"hidden_channels_{arch}"] % gnn["heads"]:
            raise SystemExit(f"The hidden channels of {arch} must be a multiple of the attention heads.")
    return jsonable(params)
