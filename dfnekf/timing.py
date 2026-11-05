"""Timing helpers shared by ``scripts/cost.py`` and ``scripts/jacobian_check.py``.

The cost figures of the paper are medians over many single calls of a jitted function (Python dispatch included), measured on one CPU thread on an
otherwise idle machine. These helpers time such calls and record the state of the machine: the number of other busy processes (needs ``psutil``;
skipped when it is not installed), the platform and the library versions. Nothing here enforces idleness: ``idle_warning`` only prints a warning.
"""
import platform
import time
import numpy as np


def other_busy_processes(threshold=5.0, interval=1.0):
    """Number of OTHER processes using more than ``threshold`` percent of one core over ``interval`` seconds; None if psutil is not installed."""
    try:
        import os
        import psutil
    except ImportError:
        return None
    me = os.getpid()
    procs = []
    for p in psutil.process_iter(["pid"]):
        try:
            if p.pid != me: p.cpu_percent(None); procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(interval)
    n = 0
    for p in procs:
        try:
            if p.cpu_percent(None) > threshold: n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def idle_warning(n_other=None):
    """Print the reminder that timings are only meaningful on an idle machine; returns the number of busy processes (None if unknown)."""
    if n_other is None: n_other = other_busy_processes()
    print("WARNING: timings are valid only on an otherwise idle machine, with one thread (one thread; do not pin the process to a core: XLA helper threads share the core and the step doubles).", flush=True)
    if n_other is None: print("  (psutil is not installed: the number of other busy processes was not recorded)", flush=True)
    elif n_other > 0: print(f"  {n_other} other process(es) use more than 5 % of a core right now: the timings below are NOT reliable", flush=True)
    else: print("  no other busy process detected", flush=True)
    return n_other


def machine_info():
    """Platform description recorded next to the timings."""
    import os
    import jax
    return dict(platform=platform.platform(), processor=platform.processor() or "n/a", cpu_count=os.cpu_count(), python=platform.python_version(), jax=jax.__version__,
                xla_flags=os.environ.get("XLA_FLAGS", ""))


def time_calls(f, args, n, warmup=2):
    """Wall time [ms] of ``n`` calls of ``f(*args)`` (each followed by ``block_until_ready``), after ``warmup`` untimed calls. Returns an array of n values."""
    import jax
    for _ in range(warmup): jax.block_until_ready(f(*args))
    t = np.empty(n)
    for i in range(n):
        a = time.perf_counter(); jax.block_until_ready(f(*args)); t[i] = time.perf_counter() - a
    return t * 1e3


def stats(t):
    """Median, p99, p99.9, worst and worst/median of an array of times."""
    t = np.asarray(t, float); med = float(np.median(t))
    return dict(median_ms=med, p99_ms=float(np.quantile(t, .99)), p999_ms=float(np.quantile(t, .999)), worst_ms=float(t.max()), worst_over_median=float(t.max() / med), n_calls=int(len(t)))


def scan_mean_ms(step, carry, xs, n):
    """Mean time per step [ms] of ``n`` steps run inside one compiled ``lax.scan`` (amortized cost without per-call dispatch); one warm run, one timed run."""
    import jax
    f = jax.jit(lambda c: jax.lax.scan(step, c, xs)[0])
    jax.block_until_ready(f(carry)); a = time.perf_counter(); jax.block_until_ready(f(carry))
    return (time.perf_counter() - a) / n * 1e3
