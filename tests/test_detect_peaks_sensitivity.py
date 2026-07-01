"""Regression test for the radial_profile / prime_log_sse crash bug.

The real, confirmed bug was `_get_shell_map`'s in-place broadcast (`r2 += g**2` on a
sparse meshgrid), which made `radial_profile` raise on every 3D field, so
`prime_log_sse`'s broad try/except silently returned the 999/0-peak fallback. That
faked "0 peaks" across the whole investigation. This guards against its return.

NOTE: peak-COUNT behaviour of the full pipeline is NOT asserted here. The pipeline
(k^2 power weighting + detect_peaks) does not reproduce historical peak counts
(e.g. a field with 40 historical peaks now scores 0-1) and is considered UNRELIABLE
for structure claims pending reconciliation with a validated reference. Do not encode
synthetic peak-count expectations as ground truth -- they proved non-representative.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quantulemapper_real as qm


def test_prime_log_sse_no_silent_fallback_on_3d():
    rho = np.abs(np.random.default_rng(0).standard_normal((32, 32, 32))) + 1.0
    d = qm.prime_log_sse(rho)
    assert not d.get("failure_reason_main"), (
        "prime_log_sse fell back due to exception: " + str(d.get("failure_reason_main"))
    )
