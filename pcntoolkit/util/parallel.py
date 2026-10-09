"""Run one task per response variable in parallel worker processes.

The functions here never change the thread settings of the calling process.
With ``n_jobs=1`` the tasks run in the calling process, one at a time, as
before.
"""

from __future__ import annotations

import numbers
import os
import warnings
from typing import Any, Callable, Iterable, Iterator

from joblib import Parallel, cpu_count, delayed, parallel_config
from threadpoolctl import threadpool_limits

from pcntoolkit.util.output import Output, Warnings

# Environment variables that state how many CPUs the user allocated:
# SLURM, Torque (PBS_NUM_PPN), PBS Pro / OpenPBS (NCPUS) and OpenMP.
ALLOCATION_ENV_VARS = ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "NCPUS", "OMP_NUM_THREADS")

# Output class settings that worker processes copy from the parent.
_OUTPUT_SETTINGS = ("_show_messages", "_show_warnings", "_show_pid", "_show_timestamp")


def allocated_cpus() -> int:
    """Return the number of CPUs this process may use.

    This is the smallest of: the CPUs this process can run on (from
    ``joblib.cpu_count``: CPU affinity, and a cgroup CPU quota when it is
    visible at the root of ``/sys/fs/cgroup``, as in containers) and the
    values of ``SLURM_CPUS_PER_TASK``, ``PBS_NUM_PPN``, ``NCPUS`` and
    ``OMP_NUM_THREADS``. For a list such as ``OMP_NUM_THREADS=4,1`` the
    first value is used. Variables that are not set, or that are not a
    positive integer, are ignored.

    Returns
    -------
    int
        Number of allocated CPUs, at least 1.
    """
    limits = [cpu_count(only_physical_cores=False)]
    for name in ALLOCATION_ENV_VARS:
        try:
            value = int(os.environ.get(name, "").split(",")[0].strip())
        except ValueError:
            continue
        if value > 0:
            limits.append(value)
    return max(1, min(limits))


def check_n_jobs(n_jobs: int) -> int:
    """Check that ``n_jobs`` is a non-zero integer and return it as ``int``.

    Parameters
    ----------
    n_jobs : int
        Requested number of worker processes. NumPy integers are accepted.

    Returns
    -------
    int
        The same value as a Python ``int``.

    Raises
    ------
    ValueError
        If ``n_jobs`` is 0, a bool or not an integer.
    """
    is_int = isinstance(n_jobs, numbers.Integral) and not isinstance(n_jobs, bool)
    if not is_int or n_jobs == 0:
        raise ValueError(f"n_jobs must be a non-zero integer, got {n_jobs!r}.")
    return int(n_jobs)


def resolve_n_jobs(n_jobs: int) -> int:
    """Convert an ``n_jobs`` argument to a number of worker processes.

    Negative values count back from the allocated CPUs, as in joblib:
    ``-1`` means all allocated CPUs, ``-2`` all but one. A positive value
    larger than the allocated CPUs is reduced to the allocated CPUs, with a
    warning.

    Parameters
    ----------
    n_jobs : int
        Requested number of worker processes. Must not be 0.

    Returns
    -------
    int
        Number of worker processes, from 1 to the allocated CPUs.

    Raises
    ------
    ValueError
        If ``n_jobs`` is 0 or not an integer.
    """
    n_jobs = check_n_jobs(n_jobs)
    allocated = allocated_cpus()
    if n_jobs < 0:
        return max(1, allocated + 1 + n_jobs)
    if n_jobs > allocated:
        Output.warning(
            Warnings.N_JOBS_ABOVE_ALLOCATION, n_jobs=n_jobs, allocated=allocated
        )
        return allocated
    return n_jobs


def _call_in_worker(
    func: Callable[..., Any], args: tuple, output_settings: dict[str, bool]
) -> tuple[Any, list[warnings.WarningMessage]]:
    """Call ``func(*args)`` in a worker with 1 BLAS/OpenMP thread.

    The worker uses the parent's ``Output`` settings. Warnings are recorded
    and returned, so that the parent can show them.
    """
    for name, value in output_settings.items():
        setattr(Output, name, value)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        result = func(*args)
    return result, caught


def map_tasks(
    func: Callable[..., Any], tasks: Iterable[tuple], n_tasks: int, n_jobs: int
) -> Iterator[Any]:
    """Yield ``func(*args)`` for each ``args`` in ``tasks``, in order.

    With one worker (``n_jobs=1`` or one task), each task is built, run in
    the calling process and yielded before the next task is built, so only
    one task is in memory at a time. The thread settings of the calling
    process are not changed.

    With more workers, the tasks run in worker processes with 1 BLAS/OpenMP
    thread each, so the total number of threads is not more than
    ``n_jobs``. At most ``2 * n_jobs`` tasks wait in the queue at a time.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to call. With more than one worker, it and its arguments
        must be picklable.
    tasks : Iterable[tuple]
        One tuple of positional arguments per call. Can be a generator.
    n_tasks : int
        Number of tasks in ``tasks``.
    n_jobs : int
        Number of worker processes, already resolved with ``resolve_n_jobs``.

    Yields
    ------
    Any
        The return values, in the order of ``tasks``.
    """
    n_workers = min(n_jobs, n_tasks)
    if n_workers <= 1:
        for args in tasks:
            yield func(*args)
        return
    output_settings = {name: getattr(Output, name) for name in _OUTPUT_SETTINGS}
    # The backend is fixed when Parallel is created, so the config does not
    # stay active in this process while the generator is paused.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        runner = Parallel(n_jobs=n_workers, return_as="generator")
    for result, caught in runner(
        delayed(_call_in_worker)(func, args, output_settings) for args in tasks
    ):
        for w in caught:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        yield result
