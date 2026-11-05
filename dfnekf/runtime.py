"""Process setup shared by the entry points: single-threaded numerics and optional CPU pinning, portable across Linux, macOS and Windows.

``single_thread()`` must be called before ``import jax`` (it sets environment variables read at import time). ``pin_to_core`` is optional and must NOT be used for timings (XLA helper threads share the pinned core: the step at N = 653 doubles, 15.6 ms instead of 7.0 ms):
it can be used for long accuracy runs (not for timings), and does nothing (returns False) where the operating system has no affinity call (macOS).
"""
import os
import sys


def single_thread():
    """Restrict XLA/Eigen and the BLAS libraries to one thread; call before importing jax."""
    flag = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    cur = os.environ.get("XLA_FLAGS", "")
    if "intra_op_parallelism_threads" not in cur: os.environ["XLA_FLAGS"] = (cur + " " + flag).strip()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")


def pin_to_core(core):
    """Pin the current process to one CPU core. Returns True if the affinity was set and verified, False if unsupported (macOS)."""
    if core is None: return False
    if hasattr(os, "sched_setaffinity"):                              # Linux
        os.sched_setaffinity(0, {int(core)})
        if os.sched_getaffinity(0) != {int(core)}: raise OSError("affinity not applied")
        return True
    if sys.platform == "win32":                                       # Windows: types must be declared, otherwise the pseudo-handle is truncated
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]; k.SetProcessAffinityMask.restype = wintypes.BOOL
        k.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]; k.GetProcessAffinityMask.restype = wintypes.BOOL
        h = k.GetCurrentProcess()
        if not k.SetProcessAffinityMask(h, 1 << int(core)): raise OSError(f"SetProcessAffinityMask failed: {ctypes.get_last_error()}")
        pm, sm = ctypes.c_size_t(), ctypes.c_size_t(); k.GetProcessAffinityMask(h, ctypes.byref(pm), ctypes.byref(sm))
        if pm.value != (1 << int(core)): raise OSError(f"affinity mask is {pm.value}, expected {1 << int(core)}")
        return True
    return False                                                     # macOS: no public per-thread affinity API


def setup(core=None, quiet=False):
    """single_thread() plus optional pinning; returns the pinning outcome."""
    single_thread(); ok = pin_to_core(core)
    if core is not None and not quiet: print(f"pin to core {core}: {'ok' if ok else 'not supported on this platform'}", flush=True)
    return ok
