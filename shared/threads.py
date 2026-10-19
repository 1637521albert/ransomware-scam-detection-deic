# Single-threaded numerical libraries for the scripts that run one worker process per
# core (expansions, feature extraction, tabular models).
#
# Every worker running multithreaded BLAS/OpenMP would start one thread per core, so
# the processes would compete for the cores (cores x cores threads), and an OpenMP
# runtime started before forking the workers can hang them. Importing this module
# sets the thread variables, which only act on the libraries loaded afterwards, so it
# is imported before numpy; limit_threads() also limits, with threadpoolctl, the
# libraries that are already loaded.

import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

for _variable in THREAD_VARIABLES:
    os.environ.setdefault(_variable, "1")


def limit_threads():
    """Limits the thread pools of the numerical libraries already loaded to one thread."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)
