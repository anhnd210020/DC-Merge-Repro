"""Inference for Experiment 8.

* Victims are the unit of inference. A victim-level statistic z_v is computed per contrast.
* `sign_flip_test`: exact one- or two-sided test of "z_v symmetric about 0" by enumerating all
  2^n sign patterns (random patterns with a fixed seed when n > 20).
* `iut`: intersection-union test, p = max of the component p-values (valid, no adjustment).
* `fixed_sequence`: hierarchical testing at alpha in a pre-specified order, stopping at the
  first non-rejection (controls the familywise error rate at alpha).
* `holm`: Holm's step-down procedure.
* `percentile_ci`: percentile interval of bootstrap draws.
"""
from itertools import product

import numpy as np


def sign_flip_test(z, alternative="greater", max_exact=20, n_random=100000, seed=0):
    z = np.asarray([x for x in z if x is not None and np.isfinite(x)], dtype=float)
    n = len(z)
    if n == 0:
        return {"n": 0, "T": None, "p": None, "exact": None, "min_attainable_p": None}
    T = float(z.mean())
    if n <= max_exact:
        signs = np.array(list(product([1.0, -1.0], repeat=n)))
        exact = True
    else:
        signs = np.random.RandomState(seed).choice([1.0, -1.0], size=(n_random, n))
        signs[0] = 1.0
        exact = False
    Ts = (signs * z).mean(1)
    eps = 1e-12 * max(1.0, abs(T))
    if alternative == "greater":
        p = float(np.mean(Ts >= T - eps))
    elif alternative == "less":
        p = float(np.mean(Ts <= T + eps))
    elif alternative == "two-sided":
        p = float(np.mean(np.abs(Ts) >= abs(T) - eps))
    else:
        raise ValueError(alternative)
    return {"n": n, "T": T, "p": p, "exact": exact, "alternative": alternative,
            "min_attainable_p": 2.0 ** (-n) if alternative != "two-sided" else 2.0 ** (1 - n)}


def iut(*ps):
    ps = [p for p in ps]
    if any(p is None for p in ps):
        return None
    return float(max(ps))


def fixed_sequence(ordered, alpha=0.05):
    """ordered: list of (name, p). Returns {name: verdict} in order."""
    out, open_ = {}, True
    for name, p in ordered:
        if not open_:
            out[name] = "not tested (sequence stopped earlier); exploratory"
        elif p is not None and p <= alpha:
            out[name] = "confirmed"
        else:
            out[name] = "not confirmed"
            open_ = False
    return out


def holm(pvals, alpha=0.05):
    """pvals: dict name -> p. Returns dict name -> (adjusted p, rejected)."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m = len(items)
    res, running, stop = {}, 0.0, False
    for i, (p, k) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        rej = (not stop) and p <= alpha / (m - i)
        if not rej:
            stop = True
        res[k] = (running, rej)
    for k, p in pvals.items():
        if p is None:
            res[k] = (None, False)
    return res


def percentile_ci(draws, level=0.95):
    x = np.asarray([d for d in draws if d is not None and np.isfinite(d)], dtype=float)
    if len(x) == 0:
        return [None, None]
    a = (1 - level) / 2
    return [float(np.percentile(x, 100 * a)), float(np.percentile(x, 100 * (1 - a)))]


def ratio_of_sums(num, den):
    num, den = np.sum(num), np.sum(den)
    return float(num / den) if abs(den) > 1e-12 else None


def multinomial_counts(rng, n):
    """Image weights of one nonparametric bootstrap resample of n images."""
    return rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
