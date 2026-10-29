"""Tests for Experiment 8. CPU only, no downloads, a few minutes.

Unit tests
  1. TSV-M and Iso-CTS equal the official code on random tensors.
  2. Harsanyi / Shapley identities, including the amplification identity kappa_M = d(EM)+d(ML)+d(EML).
  3. Statistics: exact sign-flip p-values, intersection-union, fixed sequence, Holm.
  4. Hybrids and the spectrum- and norm-matched random perturbation.
  5. Linear probes recover a linearly separable label.
End-to-end (tiny synthetic CLIP vision model, 5 tasks, 6 blocks)
  6. Positive control: two "harmer" tasks share a planted component in the middle blocks (2-3).
     Every step runs; depthstats must give the middle third the largest depth share.
     Also: depthstats' observed numbers equal factorial.json; resumability; config-hash refusal.
  7. Negative control: near-zero task updates -> no eligible pairs -> depthstats says so.

Run:  python -m pytest -q tests/test_exp8.py     (or: python tests/test_exp8.py)
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from exp8 import config as C  # noqa: E402
from exp8.harm import decompose, shapley_from_value, subsets  # noqa: E402
from exp8.operators import (SVDCache, hybrid_sel, in_units, matched_random_like, merge_iso_cts,  # noqa: E402
                            merge_tsvm)
from exp8.stats import fixed_sequence, holm, iut, sign_flip_test  # noqa: E402
import official_reference as ref  # noqa: E402


# ---------------------------------------------------------------- unit tests
def _random_tvs(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    names = ["layers.0.q.weight", "layers.0.fc1.weight", "pos.weight", "layers.0.q.bias", "conv.weight"]
    shapes = [(24, 24), (48, 24), (7, 24), (24,), (8, 3, 4, 4)]
    tvs = [{n: torch.randn(s, generator=g) for n, s in zip(names, shapes)} for _ in range(T)]
    return names, tvs


def test_operators_match_official():
    for T in (1, 2, 3, 4):
        names, tvs = _random_tvs(T, seed=T)
        objs = [SimpleNamespace(vector=tv) for tv in tvs]
        cfg = SimpleNamespace(DATASETS=list(range(T)), device="cpu",
                              method=SimpleNamespace(common_space_fraction=0.8))
        off_iso = ref.iso_cts(objs, cfg)
        off_tsv = ref.compute_and_sum_svd_mem_reduction(objs, cfg)
        ours_iso = merge_iso_cts(tvs, names, 0.8)
        ours_tsv = merge_tsvm(tvs, names)
        cache = SVDCache({i: tv for i, tv in enumerate(tvs)})
        ours_tsv_cached = merge_tsvm(tvs, names, task_ids=list(range(T)), svd_cache=cache)
        for n in names:
            assert torch.equal(off_iso[n], ours_iso[n]), f"Iso-CTS differs at {n}, T={T}"
            assert torch.equal(off_tsv[n], ours_tsv[n]), f"TSV-M differs at {n}, T={T}"
            assert torch.allclose(off_tsv[n], ours_tsv_cached[n], atol=1e-5), f"cached TSV-M {n}"


def test_decomposition_and_depth_identities():
    rng = np.random.RandomState(0)
    N = ["a", "b", "c"]
    f = {T: float(rng.uniform(40, 90)) for T in subsets(N)}
    r = decompose(f, N)
    assert abs(r["check_sum"]) < 1e-9
    assert abs(sum(r["shapley"].values()) - r["H"]) < 1e-9
    assert abs(r["H"] - r["sum_singles"] - r["interaction"]) < 1e-9
    # depth factorial: u(G) over thirds
    T3 = C.THIRDS
    u = {G: (0.0 if not G else float(rng.uniform(-5, 20))) for G in subsets(T3)}
    d, phi = shapley_from_value(u, T3)
    full = frozenset(T3)
    assert abs(sum(phi.values()) - u[full]) < 1e-9                      # efficiency
    c_M = u[frozenset(["M"])]
    r_M = u[full] - u[frozenset(["E", "Lt"])]
    kappa = r_M - c_M
    assert abs(kappa - (d[frozenset("E M".split())] + d[frozenset(["M", "Lt"])] + d[full])) < 1e-9
    # additive value: Shapley equals the single effects
    w = {"E": 1.0, "M": 7.0, "Lt": 2.0}
    d2, phi2 = shapley_from_value({G: sum(w[g] for g in G) for G in subsets(T3)}, T3)
    assert all(abs(phi2[g] - w[g]) < 1e-12 for g in T3)


def test_statistics():
    t = sign_flip_test([1.0, 2.0, 0.5, 3.0, 1.5])
    assert t["exact"] and abs(t["p"] - 1 / 32) < 1e-12 and t["min_attainable_p"] == 1 / 32
    t = sign_flip_test([1.0, -2.0, 0.5, 3.0, 1.5], alternative="two-sided")
    assert 0 < t["p"] <= 1
    # exact p by brute force for a mixed-sign case
    z = np.array([1.0, -0.4, 0.7, 2.0])
    T = z.mean()
    cnt = sum((np.array(s) * z).mean() >= T - 1e-12
              for s in np.array(np.meshgrid(*[[1, -1]] * 4)).T.reshape(-1, 4))
    assert abs(sign_flip_test(z)["p"] - cnt / 16) < 1e-12
    assert iut(0.01, 0.04) == 0.04 and iut(0.01, None) is None
    v = fixed_sequence([("a", 0.01), ("b", 0.2), ("c", 0.001)])
    assert v["a"] == "confirmed" and v["b"] == "not confirmed" and v["c"].startswith("not tested")
    h = holm({"x": 0.01, "y": 0.04})
    assert h["x"][1] and h["y"][1] and abs(h["x"][0] - 0.02) < 1e-12
    h = holm({"x": 0.03, "y": 0.04})
    assert not h["x"][1] and not h["y"][1]


def test_hybrid_and_matched_random():
    units = {"a": "pre", "b": 0, "c": 1, "d": "post"}
    kinds = {"a": "other", "b": "attn", "c": "mlp", "d": "other"}
    A = {n: torch.full((2,), 1.0) for n in units}
    B = {n: torch.full((2,), 0.0) for n in units}
    h = hybrid_sel(A, B, units, kinds, in_units([0, "post"]))
    assert [float(h[n][0]) for n in "abcd"] == [0, 1, 0, 1]
    h = hybrid_sel(A, B, units, kinds, in_units([0, 1], kind="mlp"))
    assert [float(h[n][0]) for n in "abcd"] == [0, 0, 1, 0]
    g = torch.Generator().manual_seed(0)
    D = {"m": torch.randn(20, 12, generator=g), "v": torch.randn(12, generator=g)}
    P, se, ne = matched_random_like(D, seed=5)
    assert se < 1e-6 and ne < 1e-6
    assert torch.allclose(torch.linalg.svdvals(P["m"].double()), torch.linalg.svdvals(D["m"].double()), atol=1e-5)
    assert not torch.allclose(P["m"], D["m"])
    P2, _, _ = matched_random_like(D, seed=5)
    assert torch.equal(P["m"], P2["m"]) and torch.equal(P["v"], P2["v"])   # deterministic


def test_probe_recovers_separable_label():
    from exp8.probes import fit_probe
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(3, 10, generator=g) * 4
    y = torch.randint(0, 3, (300,), generator=g)
    X = centers[y] + torch.randn(300, 10, generator=g)
    lam, corr = fit_probe(X[:200], y[:200].numpy(), X[200:], y[200:].numpy(), 3, [1e-3, 1e-1], 0.2, 100)
    assert corr.mean() > 0.95 and lam in (1e-3, 1e-1)


# ---------------------------------------------------------------- end-to-end
TASKS5 = C.TASKS[:5]
HARMERS = TASKS5[3:]           # the two tasks that carry the planted mid-depth component
MID_BLOCKS = (2, 3)            # = the middle third of 6 blocks


def _build_synthetic(root, n_img=200, n_cls=5, layers=6, scale=0.03, planted=1.0):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    from exp8.data import CLIP_MEAN, CLIP_STD, make_splits
    torch.manual_seed(0)
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                           num_attention_heads=4, image_size=224, patch_size=32, projection_dim=16)
    base = CLIPVisionModel(cfg).eval()
    base_dir = os.path.join(root, "models", "base")
    base.save_pretrained(base_dir)
    proj = torch.randn(16, 32) / 6
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    data_dir = os.path.join(root, "data")
    shared = torch.Generator().manual_seed(999)
    planted_dir = {n: torch.randn(p.shape, generator=shared) / p.shape[1] ** 0.5
                   for n, p in base.named_parameters() if p.dim() == 2}
    heads = {}
    for i, t in enumerate(TASKS5):
        ex = CLIPVisionModel(cfg).eval()
        ex.load_state_dict(base.state_dict())
        g = torch.Generator().manual_seed(100 + i)
        with torch.no_grad():
            for n, p in ex.named_parameters():
                if p.dim() == 2:
                    a = torch.randn(p.shape[0], 3, generator=g)
                    b = torch.randn(3, p.shape[1], generator=g)
                    p.add_(scale * (a @ b) / p.shape[1] ** 0.5)
                    if planted and t in HARMERS and any(f"layers.{l}." in n for l in MID_BLOCKS):
                        p.add_(planted * planted_dir[n])
                else:
                    p.add_(0.3 * scale * torch.randn(p.shape, generator=g))
        ex.save_pretrained(os.path.join(root, "models", t))
        # piecewise-constant 7x7 colour grids (one colour per 32x32 patch): diverse features
        grid = np.random.RandomState(i).randint(0, 256, size=(n_img, 3, 7, 7)).astype(np.uint8)
        imgs = np.ascontiguousarray(grid.repeat(32, axis=2).repeat(32, axis=3))
        with torch.no_grad():
            x = (torch.from_numpy(imgs).float() / 255 - mean) / std
            f = ex(pixel_values=x).pooler_output @ proj.T
            f = f / f.norm(dim=-1, keepdim=True)
            # heads = centred expert features of n_cls prototype images, so that the classes
            # depend on what the expert computes (not on the shared mean feature direction)
            protos = torch.randperm(n_img, generator=g)[:n_cls]
            heads[t] = torch.nn.functional.normalize(f[protos] - f.mean(0, keepdim=True), dim=-1)
            labels = (100.0 * f @ heads[t].T).argmax(-1).numpy()
        d = os.path.join(data_dir, t)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "images.npy"), imgs)
        np.save(os.path.join(d, "labels.npy"), labels.astype(np.int64))
        with open(os.path.join(d, "splits.json"), "w") as fh:
            json.dump(make_splits(n_img), fh)
    s = os.path.join(root, "results", "_shared")
    os.makedirs(s, exist_ok=True)
    torch.save({"heads": heads, "proj": proj, "logit_scale": 100.0}, os.path.join(s, "heads.pt"))
    return base_dir, os.path.join(root, "models", "{task}"), data_dir


def _n_evals(out, op):
    return sum(len(fs) for _, _, fs in os.walk(os.path.join(out, op, "evals")))


def _common(root, base, pat, data):
    out = os.path.join(root, "results")
    return out, ["--out", out, "--data-dir", data, "--backbone", "custom", "--base-id", base,
                 "--ft-pattern", pat, "--tasks", *TASKS5, "--device", "cpu", "--batch-size", "64",
                 "--ref-tol", "1000", "--bootstrap", "200"]


def test_end_to_end_positive_control(monkeypatch):
    from exp8.run import main
    from exp8.store import read_json
    monkeypatch.setitem(C.DEPTH, "min_victims", 2)   # 5 synthetic tasks; exercise the test path
    root = tempfile.mkdtemp(prefix="exp8pos_")
    try:
        base, pat, data = _build_synthetic(root)
        out, common = _common(root, base, pat, data)
        main(["check", *common])
        chk = read_json(os.path.join(out, "_shared", "check.json"))
        assert chk["pass"] and chk["determinism_identical"] and chk["block_runner_rel_error"] < 1e-5
        for e in chk["assets"].values():
            assert abs(e["expert"] - 100.0) < 1e-6
        main(["alpha", "--op", "ta", "--alpha-grid", "0.5", *common])
        main(["all", "--op", "ta", *common])
        for op in ("iso_cts", "tsvm"):
            main(["all", "--op", op, "--alpha", "1.0", "--k-per-victim", "1", *common])
        sel = read_json(os.path.join(out, "ta", "selection.json"))
        assert sel["n_victims"] >= 2, sel["ineligible_victims"]
        # the planted pair of harmers drives the largest harms
        top = max(sel["pairs"], key=lambda p: p["gamma_sel"])
        assert set(HARMERS) <= set(top["harmers"]), top
        fac = read_json(os.path.join(out, "ta", "factorial.json"))
        assert fac["n_layers"] == 6
        # resumability
        before = _n_evals(out, "ta")
        for s in ("subsets", "factorial", "blocks", "random"):
            main([s, "--op", "ta", *common])
        assert _n_evals(out, "ta") == before
        # analysis
        main(["select", "--all-ops", *common])
        main(["depthstats", *common])
        res = read_json(os.path.join(out, "depth_summary.json"))
        sig = res["per_operator"]["ta"]["estimates"]["sigma"]
        assert sig["M"] > sig["E"] and sig["M"] > sig["Lt"], sig
        assert os.path.exists(os.path.join(out, "who_summary.md"))
        for h in C.HYPOTHESIS_ORDER:
            assert res["confirmatory"][h]["verdict"]
        # depthstats' observed statistics match factorial.json (two independent code paths)
        from exp8.depthstats import PairData
        from exp8.run import make_runner, build_parser
        a = build_parser().parse_args(["depthstats", *common])
        a.check_svd_ops = True
        r = make_runner(a)
        for i, p in enumerate(r.pairs("ta")):
            st = PairData(r, "ta", dict(p, _index=i), 6, C.DEPTH["random_seeds"]).stats()
            row = fac["pairs"][p["id"]]
            assert abs(st["Gamma"] - row["Gamma"]) < 1e-9
            assert all(abs(st["phi"][g] - row["phi"][g]) < 1e-9 for g in C.THIRDS)
        # exploratory single-operator run writes a separate, labelled file
        main(["depthstats", "--op", "ta", "--exploratory", *common])
        assert os.path.exists(os.path.join(out, "depth_summary_exploratory_ta.md"))
        # config-hash lock
        monkeypatch.setattr(C, "config_sha256", lambda: "changed")
        with pytest.raises(SystemExit):
            main(["depthstats", *common])
        monkeypatch.undo()
        monkeypatch.setitem(C.DEPTH, "min_victims", 2)
        # replication path (same synthetic backbone stands in for ViT-L/14)
        os.makedirs(os.path.join(out, "vitl14", "_shared"), exist_ok=True)
        shutil.copy(os.path.join(out, "_shared", "heads.pt"), os.path.join(out, "vitl14", "_shared", "heads.pt"))
        main(["replicate", "--op", "ta", "--alpha", "0.5", *common])
        assert os.path.exists(os.path.join(out, "vitl14", "replicate_summary.md"))
        print(open(os.path.join(out, "depth_summary.md")).read())
    finally:
        if os.environ.get("EXP8_KEEP") is None:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("kept", root)


def test_end_to_end_negative_control():
    from exp8.run import main
    from exp8.store import read_json
    root = tempfile.mkdtemp(prefix="exp8neg_")
    try:
        base, pat, data = _build_synthetic(root, n_img=120, scale=1e-5, planted=0.0)
        out, common = _common(root, base, pat, data)
        main(["check", "--skip-svd-check", *common])
        for s in ("subsets", "analyze", "select"):
            main([s, "--op", "ta", "--alpha", "1.0", *common])
        sel = read_json(os.path.join(out, "ta", "selection.json"))
        assert sel["pairs"] == [] and len(sel["ineligible_victims"]) == len(TASKS5)
        main(["select", "--all-ops", *common])
        main(["depthstats", *common])
        res = read_json(os.path.join(out, "depth_summary.json"))
        assert res["feasible"] is False and res["n_victims"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"]))
