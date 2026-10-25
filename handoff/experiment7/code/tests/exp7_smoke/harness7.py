"""Experiment 7 CPU smoke test (synthetic 12-block model, numpy torch shim).

Builds Experiment 3/5/6 outputs with the REAL scripts (via tests/exp6_smoke/harness.py), then
runs every Experiment 7 subcommand against them.
usage: python harness7.py <lane_harm dir> <vision_lora_merge dir> <work dir> <stage> [args]
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

LANE, MFDIR, WORK, STAGE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
HERE = Path(__file__).resolve().parent
H6 = HERE.parent / "exp6_smoke" / "harness.py"
EXTRA = sys.argv[5:]
sys.argv = [str(H6), str(LANE), str(MFDIR), str(WORK), "none"]
MY_STAGE = STAGE
exec(compile(H6.read_text().split('if __name__ == "__main__":')[0], str(H6), "exec"))
STAGE = MY_STAGE
import os
if os.environ.get("SMOKE_D"):
    D = int(os.environ["SMOKE_D"])   # all8 needs D >= 8 x 16 = 128

EXP3R = WORK / "exp3_current"
OUT7 = WORK / "exp7"


def a7(**kw):
    base = dict(COMMON, exp3r=EXP3R, exp6=OUT, out=OUT7, contexts="core", reference="current",
                n_unlabeled=50, seed=0, check_agree_vectors=8, agree_tol_images=0, recompute_experts=False, all8_alpha=0.4)
    base.update(kw)
    for k in ("exp3r", "exp6", "out"):
        base[k] = Path(base[k])
    return Namespace(**base)


def build_inputs():
    import run_agreement as ra
    import rebaseline_exp3 as rb
    import run_layer_oracle as rl
    stage_exp3(list(ro.CONTEXTS)[:7])
    stage_exp5()
    ra.cmd_check(Namespace(**COMMON, exp3=EXP3, exp5=EXP5, out=OUT, contexts="core", n_unlabeled=50, seed=0,
                           max_vectors=0, skip_expert_check=False))
    rb.cmd_run(Namespace(**COMMON, exp3=EXP3, out=EXP3R, contexts="four", max_archive_diff=0.5))
    for c in ("core", "four"):
        ra.cmd_run(Namespace(**COMMON, exp3=EXP3R, exp5=EXP5, out=OUT, contexts=c, n_unlabeled=50, seed=0,
                             max_passes=2, skip_expert_check=False))
    rl.cmd_mechanism(Namespace(**COMMON, exp3=EXP3R, out=OUT, contexts="core", grouping="depth", boost=1.5))
    rl.cmd_oracle(Namespace(**COMMON, exp3=EXP3R, out=OUT, contexts="four", grouping="depth", max_passes=2))


if __name__ == "__main__":
    import run_exp7 as r7
    r7.ro.load_runtime = fake_load_runtime
    if STAGE == "inputs":
        build_inputs()
    elif STAGE == "summarize":
        r7.cmd_summarize(Namespace(out=OUT7))
    else:
        cmd, ctxs = STAGE, (EXTRA[0] if EXTRA else "core")
        kw = {}
        for extra in EXTRA[1:]:
            k, v = extra.split("=", 1)
            kw[k] = (v == "1") if k == "recompute_experts" else (int(v) if k == "agree_tol_images" else v)
        args = a7(contexts=ctxs, **kw)
        args.out.mkdir(parents=True, exist_ok=True)
        r7.record_config(args)
        {"replay": r7.cmd_replay, "checks": r7.cmd_checks, "gate": r7.cmd_gate, "controls": r7.cmd_controls,
         "method": lambda a: r7.cmd_proxy_search(a, "method"), "entropy": lambda a: r7.cmd_proxy_search(a, "entropy"),
         "wide": r7.cmd_wide, "all8": lambda a: r7.cmd_proxy_search(a, "all8")}[cmd](args)
