"""Print one line per step of every LP604765 test: duration, sampling, mean current, Ah, V start/end."""
import sys

import numpy as np

from lpdata import list_tests, load_test

tests = sys.argv[1:] or list_tests()
for test in tests:
    d = load_test(test)
    print("=== %s  (%.2f h, %d samples)" % (test, d["t"][-1] / 3600, len(d["t"])))
    for k, n in enumerate(d["names"]):
        m = d["step"] == k
        if not m.any():
            print("   %-34s EMPTY" % n); continue
        t, V, I = d["t"][m], d["V"][m], d["I"][m]
        dt = np.median(np.diff(t)) if len(t) > 1 else np.nan
        ah = np.trapezoid(I, t) / 3600
        print("   %-34s %8.0f s dt %5.2f  I %7.3f [%7.3f,%7.3f]  Q %7.4f Ah  V %.3f -> %.3f [%.3f,%.3f]"
              % (n, t[-1] - t[0], dt, I.mean(), I.min(), I.max(), ah, V[0], V[-1], V.min(), V.max()))
