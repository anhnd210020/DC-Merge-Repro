"""Experiment 8 command line: where in depth is merging harm created?

python -m exp8.run <step> [--op ta|iso_cts|tsvm] --out OUT --data-dir DATA [options]

Stage A (selection half)
  prepare-data   download each task's test split once and cache it as uint8 tensors
  check          must-hold checks (assets, determinism, block runner, operator identities)
  alpha          each operator's global coefficient on random contexts
  subsets        merge every task subset of size 1-4 and evaluate its members
  analyze        neighbour decomposition of every (victim, context) pair; descriptive breadth
  select         victim-stratified selection of pairs and harmers (frozen);
                 `select --all-ops` writes the pooled selection_all.json
Stage B (report half)
  report         'who' decomposition of the selected pairs with fake-neighbour controls
  factorial      B1: 2^3 depth-third hybrids between A = M(S) and B = M(S minus harmers)
  blocks         B2: single-block / attention-vs-MLP removal and creation
  random         B3: spectrum- and norm-matched random perturbation per third
  own            B4: where the victim's own update matters
  probes         B5: linear probes of the victim's labels at every depth
  patch          B6: activation patching by block
  crossterm      B7: generation vs transport of the activation cross-term
  weightspace    B8: weight-space interference profile vs block removal
Analysis
  depthstats     B9: confirmatory tests, intervals, sensitivity, 'who' family
  replicate      B10: ViT-L/14 task-arithmetic replication into OUT/vitl14
  all            Stage A + Stage B for one operator (--op)
"""
import argparse
import copy
import os
import sys

from . import config as C

STAGE_A = ["alpha", "subsets", "analyze", "select"]
STAGE_B = ["report", "factorial", "blocks", "random", "own", "probes", "patch", "crossterm", "weightspace"]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step")
    ap.add_argument("--op", choices=C.OPERATORS, help="merging operator")
    ap.add_argument("--out", required=True, help="results folder")
    ap.add_argument("--data-dir", required=True, help="image cache folder (created by prepare-data)")
    ap.add_argument("--backbone", default="ViT-B-32", choices=list(C.BACKBONES) + ["custom"])
    ap.add_argument("--base-id", default=None, help="override pretrained model id/path")
    ap.add_argument("--ft-pattern", default=None, help="override fine-tuned id/path, with {task}")
    ap.add_argument("--tasks", nargs="*", default=None, help="override the task list (tests only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau-device", default=None, help="where task vectors live (cpu for ViT-L/14)")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--gpu-cache-gb", type=float, default=8.0, help="images kept on GPU")
    ap.add_argument("--merge-cache", type=int, default=3, help="merged models kept in memory")
    ap.add_argument("--alpha", type=float, default=None, help="skip alpha selection, use this value")
    ap.add_argument("--alpha-grid", type=float, nargs="*", default=None)
    ap.add_argument("--alpha-contexts", type=int, default=None, help=f"default {C.N_ALPHA_CONTEXTS}")
    ap.add_argument("--screen-cap", type=int, default=None, help=f"default {C.SCREEN_CAP}")
    ap.add_argument("--k-per-victim", type=int, default=None, help=f"default {C.DEPTH['k_per_victim']}")
    ap.add_argument("--all-ops", action="store_true", help="select: write the pooled selection_all.json")
    ap.add_argument("--probe-depths", choices=["all", "boundaries"], default="all")
    ap.add_argument("--skip-svd-check", action="store_true", help="check: skip the TSV-M/Iso-CTS identities")
    ap.add_argument("--bootstrap", type=int, default=None, help=f"depthstats draws (default {C.BOOTSTRAP})")
    ap.add_argument("--exploratory", action="store_true", help="depthstats --op X on one operator, labelled")
    ap.add_argument("--ref-tol", type=float, default=1.0, help="pp tolerance for the asset check")
    ap.add_argument("--force", action="store_true", help="continue even if `check` fails")
    return ap


def make_runner(args):
    from .depth import DepthSteps
    from .pipeline import Runner

    class Exp8Runner(DepthSteps, Runner):
        pass

    return Exp8Runner(args)


def run_replicate(args):
    """B10: the replication backbone, task arithmetic, into OUT/vitl14."""
    from .depthstats import replicate_summary
    op = args.op or "ta"
    src_out = args.out
    src_sel = os.path.join(src_out, op, "selection.json")
    if not os.path.exists(src_sel):
        sys.exit(f"replicate needs {src_sel}")
    a = copy.copy(args)
    a.out = os.path.join(src_out, "vitl14")
    if args.backbone == "ViT-B-32":
        a.backbone = "ViT-L-14"
    a.tau_device = args.tau_device or "cpu"
    a.alpha_contexts = args.alpha_contexts or 5
    a.screen_cap = args.screen_cap or 2000
    r = make_runner(a)
    r.step_check()
    if a.alpha is None and not os.path.exists(os.path.join(a.out, op, "alpha.json")):
        r.step_alpha(op)
    if not os.path.exists(os.path.join(a.out, op, "selection.json")):
        r.step_replicate_select(op, src_sel)
    r.step_factorial(op)
    r.step_random(op)
    replicate_summary(r, op, src_out, args.bootstrap or C.BOOTSTRAP)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.check_svd_ops = not args.skip_svd_check

    if args.step == "prepare-data":
        from .data import prepare_task
        for t in (args.tasks or C.TASKS):
            prepare_task(t, args.data_dir)
        return
    if args.step == "replicate":
        run_replicate(args)
        return

    r = make_runner(args)
    if args.step == "check":
        r.step_check()
        return
    if args.step == "depthstats":
        from .depthstats import run
        run(r, args)
        return
    if args.step == "select" and args.all_ops:
        if args.op:
            r.step_select(args.op)
        r.step_select_all()
        return
    if not args.op:
        sys.exit("--op is required for this step")
    steps = {s: getattr(r, "step_" + s) for s in STAGE_A + STAGE_B}
    if args.step == "all":
        for s in STAGE_A + STAGE_B:
            if s == "alpha" and (args.alpha is not None or
                                 os.path.exists(os.path.join(args.out, args.op, "alpha.json"))):
                continue
            steps[s](args.op)
        return
    if args.step not in steps:
        sys.exit(f"unknown step {args.step}")
    steps[args.step](args.op)


if __name__ == "__main__":
    main()
