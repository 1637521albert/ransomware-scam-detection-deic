# Ordered parallel map over forked worker processes.
#
# The workers are forked when the map starts, so they inherit the function and
# the items (and everything else in memory, such as the BlockSci chain and the
# expanded graph) without pickling them. Only the results travel back, so they
# must be picklable (plain Python values). The results keep the order of the
# items, so the output is the same as a sequential loop. Several maps can also
# share one pool (forked_pool), so its workers are forked once and keep what they
# cache; the items are then sent to them.

import gc
import math
import multiprocessing
from contextlib import contextmanager

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
    # Forked map: the function and the items were inherited
    start, end = bounds
    func, items = _TASK["func"], _TASK["items"]
    return start, [func(item) for item in items[start:end]]


def _run_sent_chunk(task):
    # Map on an existing pool: the function (by reference) and the items are sent
    start, func, items = task
    return start, [func(item) for item in items]


def _fork_pool(workers):
    # The objects that exist are moved out of the garbage collector's reach while the
    # workers are forked, so the workers do not copy the inherited memory pages just
    # by inspecting them (the workers keep them frozen; the parent unfreezes them)
    gc.freeze()
    try:
        return multiprocessing.get_context("fork").Pool(workers)
    finally:
        gc.unfreeze()


@contextmanager
def forked_pool(workers):
    """Pool of `workers` processes forked now, for several parallel_map calls (pool=...):
    the workers inherit the memory as it is when the pool is created, and what they cache
    lasts until the pool is closed."""
    pool = _fork_pool(workers)
    try:
        yield pool
        pool.close()
        pool.join()
    finally:
        pool.terminate()


def parallel_map(func, items, workers, desc=None, chunks_per_worker=16, max_chunk=256, pool=None):
    """[func(item) for item in items], computed by `workers` forked processes, or by
    the workers of `pool` (forked_pool; func must then be a module-level function and
    the items picklable, as they are sent to the workers).

    The items are sent to the workers in small chunks (at most max_chunk items), so a few
    very costly items do not keep one worker busy while the others are idle. The chunks
    are collected as they finish (the progress shows the items done) and put back in the
    order of the items."""
    items = items if isinstance(items, list) else list(items)
    n = len(items)
    if n == 0 or (pool is None and (workers <= 1 or n <= 1)):
        return [func(item) for item in tqdm(items, desc=desc, disable=desc is None)]

    chunk = max(1, min(math.ceil(n / (max(1, workers) * chunks_per_worker)), max_chunk))
    bounds = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    parts = {}

    def collect(results):
        with tqdm(total=n, desc=desc, disable=desc is None) as bar:
            for start, part in results:
                parts[start] = part
                bar.update(len(part))

    if pool is not None:
        collect(pool.imap_unordered(_run_sent_chunk, ((start, func, items[start:end]) for start, end in bounds)))
    else:
        _TASK["func"], _TASK["items"] = func, items
        try:
            with forked_pool(min(workers, len(bounds))) as own_pool:
                collect(own_pool.imap_unordered(_run_chunk, bounds))
        finally:
            _TASK.clear()
    return [result for start, _ in bounds for result in parts[start]]
