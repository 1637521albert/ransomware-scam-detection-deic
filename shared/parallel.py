# Ordered parallel map over forked worker processes.
#
# The workers are forked when the map starts, so they inherit the function and
# the items (and everything else in memory, such as the BlockSci chain and the
# expanded graph) without pickling them. Only the results travel back, so they
# must be picklable (plain Python values). The results keep the order of the
# items, so the output is the same as a sequential loop.

import gc
import math
import multiprocessing
import os

from tqdm import tqdm

_TASK = {}


def available_memory_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def blocksci_workers(requested="auto"):
    """Workers of a BlockSci (disk-bound) step: the requested number, or with "auto"
    BLOCKSCI_IO_FACTOR per core, limited by the available memory."""
    import config
    if requested != "auto":
        return max(1, int(requested))
    workers = config.CPU_COUNT * config.BLOCKSCI_IO_FACTOR
    memory = available_memory_bytes()
    if memory:
        fit = int(0.8 * memory // (config.BLOCKSCI_WORKER_MEMORY_GB * 2 ** 30))
        workers = min(workers, max(config.CPU_COUNT, fit))
    return max(1, workers)


def _run_chunk(bounds):
    start, end = bounds
    func, items = _TASK["func"], _TASK["items"]
    return [func(item) for item in items[start:end]]


def parallel_map(func, items, workers, desc=None, chunks_per_worker=16, max_chunk=256):
    """[func(item) for item in items], computed by `workers` forked processes.

    The items are sent to the workers in small chunks (at most max_chunk items), so
    a few very costly items do not keep one worker busy while the others are idle."""
    items = items if isinstance(items, list) else list(items)
    n = len(items)
    if workers <= 1 or n <= 1:
        return [func(item) for item in tqdm(items, desc=desc, disable=desc is None)]

    chunk = max(1, min(math.ceil(n / (workers * chunks_per_worker)), max_chunk))
    bounds = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    workers = min(workers, len(bounds))

    _TASK["func"], _TASK["items"] = func, items
    # Objects created so far are moved out of the garbage collector's reach, so the
    # workers do not copy the inherited memory pages just by inspecting them
    gc.collect()
    gc.freeze()
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = []
            with tqdm(total=n, desc=desc, disable=desc is None) as bar:
                for part in pool.imap(_run_chunk, bounds):
                    parts.append(part)
                    bar.update(len(part))
    finally:
        gc.unfreeze()
        _TASK.clear()
    return [result for part in parts for result in part]
