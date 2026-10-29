"""Exact harm decompositions over neighbour subsets (Harsanyi dividends, Shapley values).

For victim v in context S with neighbours N, f(T) is v's normalized accuracy (pp) in the
merge of {v} u T. Harm with neighbours R is g(R) = f(empty) - f(R). The Harsanyi dividend of a
non-empty T subset of N is d(T) = sum_{R subset T} (-1)^{|T|-|R|} g(R); the dividends sum
exactly to the context harm H = g(N). Order-1 dividends are single-neighbour effects (what a
pairwise mergeability predictor sees); order >= 2 dividends are interactions.
"""
from itertools import combinations

import numpy as np


def subsets(items):
    items = tuple(items)
    for k in range(len(items) + 1):
        for c in combinations(items, k):
            yield frozenset(c)


def dividends(f, neighbours):
    """f: dict frozenset(T) -> value for every T subset of neighbours."""
    f0 = f[frozenset()]
    g = {T: f0 - f[T] for T in f}
    d = {}
    for T in subsets(neighbours):
        if not T:
            continue
        d[T] = sum(((-1) ** (len(T) - len(R))) * g[R] for R in subsets(T))
    return d


def shapley_from_value(u, players):
    """Exact Harsanyi dividends and Shapley values of a value function on all subsets.

    u: dict frozenset -> value for every subset of `players` (u[empty] is usually 0).
    Returns dividends {frozenset: d}, shapley {player: phi}; sum(phi) == u[all] - u[empty].
    """
    players = tuple(players)
    d = {}
    for T in subsets(players):
        if T:
            d[T] = sum(((-1) ** (len(T) - len(R))) * u[R] for R in subsets(T))
    phi = {j: sum(v / len(T) for T, v in d.items() if j in T) for j in players}
    return d, phi


def decompose(f, neighbours):
    d = dividends(f, neighbours)
    H = f[frozenset()] - f[frozenset(neighbours)]
    singles = {j: d[frozenset([j])] for j in neighbours}
    inter = sum(v for T, v in d.items() if len(T) >= 2)
    by_order = {}
    for T, v in d.items():
        by_order[len(T)] = by_order.get(len(T), 0.0) + v
    shap = {j: sum(v / len(T) for T, v in d.items() if j in T) for j in neighbours}
    return {
        "H": H,
        "singles": singles,
        "sum_singles": sum(singles.values()),
        "interaction": inter,
        "interaction_share": (abs(inter) / abs(H)) if abs(H) > 1e-9 else None,
        "by_order": by_order,
        "shapley": shap,
        "dividends": {"+".join(sorted(T)): v for T, v in d.items()},
        "check_sum": sum(d.values()) - H,  # must be ~0 (exactness)
    }


def norm_acc(correct, expert_correct):
    e = expert_correct.mean()
    return 100.0 * correct.mean() / e if e > 0 else float("nan")


def bootstrap_decomposition(corr, expert, neighbours, B=1000, seed=0):
    """Paired image bootstrap of H, interaction share and sum of singles.

    corr: dict frozenset(T) -> bool array (same images for every T); expert: bool array.
    """
    n = len(expert)
    rng = np.random.RandomState(seed)
    Hs, shares, singles, inters = [], [], [], []
    keys = list(corr)
    mats = np.stack([corr[k] for k in keys]).astype(np.float32)
    for _ in range(B):
        idx = rng.randint(0, n, n)
        e = expert[idx].mean()
        accs = mats[:, idx].mean(1)
        f = {k: (100.0 * a / e if e > 0 else np.nan) for k, a in zip(keys, accs)}
        r = decompose(f, neighbours)
        Hs.append(r["H"])
        inters.append(r["interaction"])
        singles.append(r["sum_singles"])
        shares.append(r["interaction_share"] if r["interaction_share"] is not None else np.nan)

    def ci(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return [None, None]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {"H_ci": ci(Hs), "interaction_ci": ci(inters), "interaction_share_ci": ci(shares),
            "sum_singles_ci": ci(singles)}
