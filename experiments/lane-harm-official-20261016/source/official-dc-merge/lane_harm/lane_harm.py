"""Lane-level harm attribution for DC-Merge: per-task accounting plus causal interventions.

Built on top of the repository's own ``_dc_merge_matrix_group`` (called with
``return_intermediates=True``) so the merge analysed here *is* the production merge.

For every module and every task i, the task's update travels through a fixed
chain of operations.  The code materialises the parameter-space matrix that
task i would see after each stage and measures a data-free *readout error*
on the task's own input subspace (rows of its top singular vectors):

    e(D) = || (D - Delta_i) V_i || / || Delta_i V_i ||

Stages (each includes the merge coefficient alpha once the update is scaled):

    orig        Delta_i                                  (e = 0 by definition)
    scaled      alpha * Delta_i
    truncated   alpha * U_r S_r V_r^T
    smoothed    alpha * U_r S_hat V_r^T
    lane_proj   alpha * Ut_i [M_i]_ii Vt_i^T             own content in own lane (rotation / spill)
    lane_sparse alpha * Ut_i [S_k(M_i)]_ii Vt_i^T        own top-k
    lane_ties   alpha * Ut_i [A]_ii Vt_i^T               TIES in lane i (dilution, conflict, injection)
    merged      alpha * full merged matrix               adds other lanes read through V_i

Increments between consecutive stages telescope exactly to e(merged).  The
TIES step is further split into three disjoint coordinate classes of lane i:

    dilution   own entry kept, same sign, value averaged with companions
    conflict   own entry non-zero but out-voted (sign flipped or zeroed)
    injection  own entry zero, companions write a non-zero value

The readout error is a *diagnostic proxy*, not accuracy.  The direct test is
the intervention set below, whose merged weights you evaluate with the normal
pipeline: each removes exactly one mechanism in exactly one lane, and the
run records that nothing outside that lane changed.

Interventions (target task p):
    none                  production merge (verified equal to dc_merge)
    protect_lane          lane p := own top-k entries (removes all three classes)
    fix_conflict          restore own value on conflict coordinates of lane p
    block_injection       zero injected coordinates of lane p
    undilute              restore own value on dilution coordinates of lane p
    control_matched_lane  same kind of edit, same energy, applied to another lane q
    restore_own_update    replace lane p's contribution by p's own smoothed update
                          (no rotation, no spill, no sparsification).  This edits the
                          merged matrix in parameter space, not the aggregate; the
                          run records that the change equals exactly that swap.

Spill accounting: of task i's projected energy ||M_i||^2, the records split
the share in its own block, in other tasks' diagonal blocks (other lanes),
and in off-diagonal blocks (always removed by the structural mask).
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

MODES = ("none", "protect_lane", "fix_conflict", "block_injection", "undilute",
         "control_matched_lane", "restore_own_update")
LANE_LOCAL_MODES = MODES[:6]   # edits confined to one block of the TIES aggregate
STAGES = ("orig", "scaled", "truncated", "smoothed", "lane_proj", "lane_sparse",
          "lane_ties", "merged")
TIES_CLASSES = ("dilution", "conflict", "injection")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _fro2(x: torch.Tensor) -> float:
    return float(torch.sum(x.double() ** 2))


def _blk(offsets, i):
    s, e = offsets[i]
    return slice(s, e)


def _reference_basis(delta: torch.Tensor, max_rank: int = 16, tol: float = 1e-6):
    """Right singular vectors spanning the task's own input subspace."""
    _u, s, v = torch.linalg.svd(delta, full_matrices=False)
    if s.numel() == 0 or float(s[0]) == 0.0:
        raise ValueError("task delta is zero")
    r = int(min(max_rank, int((s > tol * s[0]).sum())))
    return v[:r, :].T, r          # n x r


def _readout(stage: torch.Tensor, delta: torch.Tensor, vref: torch.Tensor):
    """Return (numerator^2, denominator^2) so errors aggregate across modules."""
    num2 = _fro2((stage - delta) @ vref)
    den2 = _fro2(delta @ vref)
    return num2, den2


def classify_lane(own: torch.Tensor, lane: torch.Tensor):
    """Partition lane coordinates into dilution / conflict / injection masks."""
    own_nz = own != 0
    same_sign = torch.sign(lane) == torch.sign(own)
    dilution = own_nz & same_sign & (lane != own)
    conflict = own_nz & ~same_sign
    injection = (~own_nz) & (lane != 0)
    return {"dilution": dilution, "conflict": conflict, "injection": injection}


# ----------------------------------------------------------------------------
# interventions on the TIES aggregate (lane-local by construction)
# ----------------------------------------------------------------------------
def _intervene(agg, filtered, offsets, mode, target, rng):
    """Return (new_agg, info).  Only the block of the edited lane may change."""
    new = agg.clone()
    info: dict[str, Any] = {"mode": mode, "target": target}
    if mode in ("none", "restore_own_update"):
        info["edited_lane"] = None
        return new, info
    if target is None:
        raise ValueError(f"mode {mode!r} requires a target task index")

    b = _blk(offsets, target)
    own = filtered[target][b, b]
    lane = agg[b, b]
    cls = classify_lane(own, lane)

    if mode == "protect_lane":
        new[b, b] = own
    elif mode == "fix_conflict":
        new[b, b] = torch.where(cls["conflict"], own, lane)
    elif mode == "undilute":
        new[b, b] = torch.where(cls["dilution"], own, lane)
    elif mode == "block_injection":
        new[b, b] = torch.where(cls["injection"], torch.zeros_like(lane), lane)
    elif mode == "control_matched_lane":
        # Energy that protect_lane would change in lane p ...
        want = _fro2(lane - own)
        others = [q for q in range(len(offsets)) if q != target]
        avail = {q: _fro2(agg[_blk(offsets, q), _blk(offsets, q)]
                          - filtered[q][_blk(offsets, q), _blk(offsets, q)]) for q in others}
        able = [q for q in others if avail[q] >= want]
        if able:            # random lane among those that can supply matched energy
            q = able[int(rng.integers(len(able)))]
            info["control_underpowered"] = False
        elif others:        # none can: use the richest lane and flag it
            q = max(others, key=lambda x: avail[x])
            info["control_underpowered"] = True
        else:
            q = None
            info["control_underpowered"] = True
        info["control_lane"] = q
        info["target_energy"] = want
        if q is None or want == 0.0:
            info["matched_energy"] = 0.0
            info["edited_lane"] = q
            return new, info
        bq = _blk(offsets, q)
        own_q = filtered[q][bq, bq]
        lane_q = agg[bq, bq]
        diff = (lane_q - own_q).reshape(-1)
        idx = torch.nonzero(diff != 0, as_tuple=False).reshape(-1)
        perm = idx[torch.as_tensor(rng.permutation(idx.numel()), dtype=torch.long)] if idx.numel() else idx
        flat = lane_q.clone().reshape(-1)
        own_flat = own_q.reshape(-1)
        got = 0.0
        for k in perm.tolist():           # ... applied to a random other lane, same edit type
            if got >= want:
                break
            got += float(diff[k].double() ** 2)
            flat[k] = own_flat[k]
        new[bq, bq] = flat.reshape(lane_q.shape)
        info["matched_energy"] = got
        info["edited_lane"] = q
        return new, info
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    info["edited_lane"] = target
    info["changed_energy"] = _fro2(new[b, b] - lane)
    return new, info


def _locality_check(old, new, offsets, edited_lane):
    """Count entries changed outside the edited lane (must be zero)."""
    changed = old != new
    if edited_lane is not None:
        b = _blk(offsets, edited_lane)
        changed[b, b] = False
    return int(changed.sum())


# ----------------------------------------------------------------------------
# one module
# ----------------------------------------------------------------------------
def attribute_group(vecs: Sequence[torch.Tensor], group_fn: Callable, ties_fn: Callable, *,
                alpha: float, mode: str = "none", target: int | None = None,
                rng=None, dc_kwargs: Mapping[str, Any] | None = None):
    """Attribute harm in one module.  Returns (merged_matrix, per_task_records, checks)."""
    import numpy as np
    rng = rng if rng is not None else np.random.default_rng(0)
    dc_kwargs = dict(dc_kwargs or {})
    if dc_kwargs.get("task_densities") is not None:
        raise ValueError("task_densities must be None")
    merged0, inter = group_fn(list(vecs), return_intermediates=True, **dc_kwargs)
    U = inter["cover_space_u"]
    Vt = inter["cover_space_vT"]
    F = inter["filtered_matrices"]
    M = inter["projected_matrices"]
    mask = inter["structural_mask"]
    offsets = inter["rank_offsets"]
    ranks = inter["task_ranks"]
    agg = inter["ties_aggregate"]
    N = len(vecs)

    checks: dict[str, Any] = {}
    checks["ties_recompute_maxabs"] = float((ties_fn(list(F)) - agg).abs().max())
    checks["merge_recompute_maxabs"] = float(
        (torch.linalg.multi_dot((U, agg * mask, Vt)) - merged0).abs().max())

    new_agg, info = _intervene(agg, F, offsets, mode, target, rng)
    checks["outside_lane_changed_entries"] = _locality_check(agg, new_agg, offsets,
                                                             info.get("edited_lane"))
    merged = torch.linalg.multi_dot((U, new_agg * mask, Vt))
    if mode == "restore_own_update":
        b = _blk(offsets, target)
        lane_term = U[:, b] @ agg[b, b] @ Vt[b, :]
        own_update = (inter["singular_directions_u"][target]
                      @ torch.diag(inter["smoothed_singular_values"][target])
                      @ inter["singular_directions_v"][target])
        swapped = merged - lane_term + own_update
        checks["swap_residual_maxabs"] = float(
            ((swapped - merged) - (own_update - lane_term)).abs().max())
        merged = swapped

    records = []
    for i in range(N):
        delta = vecs[i]
        vref, rref = _reference_basis(delta)
        u, s, v = torch.linalg.svd(delta, full_matrices=False)
        r = ranks[i]
        b = _blk(offsets, i)
        Ub, Vb = U[:, b], Vt[b, :]
        smoothed_sv = inter["smoothed_singular_values"][i]
        su = inter["singular_directions_u"][i]
        sv = inter["singular_directions_v"][i]

        stages = OrderedDict()
        stages["orig"] = delta
        stages["scaled"] = alpha * delta
        stages["truncated"] = alpha * (u[:, :r] @ torch.diag(s[:r]) @ v[:r, :])
        stages["smoothed"] = alpha * (su @ torch.diag(smoothed_sv) @ sv)
        stages["lane_proj"] = alpha * (Ub @ M[i][b, b] @ Vb)
        stages["lane_sparse"] = alpha * (Ub @ F[i][b, b] @ Vb)
        if mode == "restore_own_update" and i == target:
            stages["lane_ties"] = stages["smoothed"]
        else:
            stages["lane_ties"] = alpha * (Ub @ new_agg[b, b] @ Vb)
        stages["merged"] = alpha * merged

        num2, den2 = {}, None
        for name, mat in stages.items():
            n2, d2 = _readout(mat, delta, vref)
            num2[name] = n2
            den2 = d2

        own = F[i][b, b]
        lane = new_agg[b, b]
        cls = classify_lane(own, lane)
        d_lane = lane - own
        parts = {c: d_lane * cls[c] for c in TIES_CLASSES}
        part_residual = float((d_lane - sum(parts.values())).abs().max())
        class_num2 = {c: _readout(alpha * (Ub @ (own + parts[c]) @ Vb), delta, vref)[0]
                      for c in TIES_CLASSES}

        # spill: where task i's own projected energy sits before top-k
        Mi = M[i]
        proj_total = _fro2(Mi)
        proj_own = _fro2(Mi[b, b])
        proj_other_lanes = sum(_fro2(Mi[_blk(offsets, j), _blk(offsets, j)])
                               for j in range(N) if j != i)
        proj_off_diag = proj_total - proj_own - proj_other_lanes

        own_lane_e2 = _fro2(own)
        by_source = [0.0 if j == i else _fro2(F[j][b, b]) for j in range(N)]
        foreign_e2 = sum(by_source)
        # entries of lane i that the TIES vote handed to each companion j
        won_by_source = []
        for j in range(N):
            if j == i:
                won_by_source.append(0)
                continue
            fj = F[j][b, b]
            won_by_source.append(int(((fj != 0) & (torch.sign(fj) == torch.sign(lane))
                                      & (lane != 0) & (own == 0)).sum()
                                     + ((fj != 0) & cls["conflict"]
                                        & (torch.sign(fj) == torch.sign(lane))).sum()))
        records.append({
            "task_index": i,
            "rank": int(r),
            "reference_rank": int(rref),
            "support_kept": int((F[i] != 0).sum()),
            "support_kept_in_own_lane": int((own != 0).sum()),
            "support_wasted_off_diagonal": int(((F[i] != 0) & (mask == 0)).sum()),
            "projected_energy": {"total": proj_total, "own_lane": proj_own,
                                 "other_lanes": proj_other_lanes, "off_diagonal": proj_off_diag},
            "own_lane_energy": own_lane_e2,
            "foreign_energy_in_lane": foreign_e2,
            "foreign_energy_by_source": by_source,
            "lane_entries_won_by_source": won_by_source,
            "class_counts": {c: int(cls[c].sum()) for c in TIES_CLASSES},
            "class_energy": {c: _fro2(parts[c]) for c in TIES_CLASSES},
            "partition_residual_maxabs": part_residual,
            "readout_num2": num2,
            "readout_den2": den2,
            "class_readout_num2": class_num2,
        })
    return merged, records, {**checks, **info}


# ----------------------------------------------------------------------------
# all modules
# ----------------------------------------------------------------------------
def _err(num2, den2):
    return math.sqrt(num2 / den2) if den2 > 0 else float("nan")


def summarise(records_by_module: Mapping[str, list], task_names: Sequence[str]):
    """Aggregate module records per task (energy-weighted, exact telescoping)."""
    out = OrderedDict()
    for i, name in enumerate(task_names):
        num = {s: 0.0 for s in STAGES}
        den = 0.0
        cnum = {c: 0.0 for c in TIES_CLASSES}
        own_e = foreign_e = 0.0
        counts = {c: 0 for c in TIES_CLASSES}
        wasted = kept = 0
        proj = {"total": 0.0, "own_lane": 0.0, "other_lanes": 0.0, "off_diagonal": 0.0}
        src_e = [0.0] * len(task_names)
        src_won = [0] * len(task_names)
        for recs in records_by_module.values():
            r = recs[i]
            for key in proj:
                proj[key] += r["projected_energy"][key]
            for j in range(len(task_names)):
                src_e[j] += r["foreign_energy_by_source"][j]
                src_won[j] += r["lane_entries_won_by_source"][j]
            for s in STAGES:
                num[s] += r["readout_num2"][s]
            den += r["readout_den2"]
            for c in TIES_CLASSES:
                cnum[c] += r["class_readout_num2"][c]
                counts[c] += r["class_counts"][c]
            own_e += r["own_lane_energy"]
            foreign_e += r["foreign_energy_in_lane"]
            wasted += r["support_wasted_off_diagonal"]
            kept += r["support_kept"]
        err = {s: _err(num[s], den) for s in STAGES}
        inc = OrderedDict((STAGES[k], err[STAGES[k]] - err[STAGES[k - 1]])
                          for k in range(1, len(STAGES)))
        lane_sparse = err["lane_sparse"]
        out[name] = {
            "readout_error": err,
            "stage_increment": inc,
            "telescoping_residual": (sum(inc.values()) - (err["merged"] - err["orig"])),
            # error added by each TIES class applied alone on top of own top-k
            "ties_class_increment": {c: _err(cnum[c], den) - lane_sparse for c in TIES_CLASSES},
            "ties_class_counts": counts,
            "spill_share": {k: (proj[k] / proj["total"] if proj["total"] > 0 else float("nan"))
                            for k in ("own_lane", "other_lanes", "off_diagonal")},
            "leakage_ratio": foreign_e / own_e if own_e > 0 else float("inf"),
            # directional: how much each companion j writes into this task's lane
            "leakage_ratio_by_source": {task_names[j]: (src_e[j] / own_e if own_e > 0 else float("inf"))
                                        for j in range(len(task_names)) if j != i},
            "lane_entries_won_by_source": {task_names[j]: src_won[j]
                                           for j in range(len(task_names)) if j != i},
            "support_kept_total": kept,
            "support_wasted_off_diagonal": wasted,
        }
    return out


def merge_with_attribution(deltas_dict: Mapping[str, Sequence[torch.Tensor]], mf, task_names: Sequence[str], *,
                alpha: float, mode: str = "none", target: str | None = None, seed: int = 0,
                dc_kwargs: Mapping[str, Any] | None = None, log_dir: str | Path | None = None,
                verify_against_dc_merge: bool = True, atol: float = 1e-5):
    """Merge every module with attribution; return (merged_deltas, summary).

    ``mf`` is the repository module ``vision_lora_merge/merging_functions.py``.
    ``target`` is a task name.  Writes ``<log_dir>/<mode>[__<target>].jsonl``
    (one line per module) and a matching ``.summary.json``.
    """
    import numpy as np
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    dc_kwargs = dict(dc_kwargs or {})
    tidx = None if target is None else list(task_names).index(target)
    rng = np.random.default_rng(seed)

    merged, recs_by_mod, checks_by_mod = OrderedDict(), OrderedDict(), OrderedDict()
    for key, vecs in deltas_dict.items():
        if len(vecs) != len(task_names):
            raise ValueError(f"module {key}: {len(vecs)} deltas for {len(task_names)} tasks")
        m, recs, chk = attribute_group(vecs, mf._dc_merge_matrix_group, mf.ties_small,
                                   alpha=alpha, mode=mode, target=tidx, rng=rng,
                                   dc_kwargs=dc_kwargs)
        merged[key], recs_by_mod[key], checks_by_mod[key] = m, recs, chk
    merged = OrderedDict(sorted(merged.items()))

    verification = {
        "max_ties_recompute_maxabs": max(c["ties_recompute_maxabs"] for c in checks_by_mod.values()),
        "max_merge_recompute_maxabs": max(c["merge_recompute_maxabs"] for c in checks_by_mod.values()),
        "total_outside_lane_changed_entries": sum(c["outside_lane_changed_entries"]
                                                  for c in checks_by_mod.values()),
        "max_partition_residual_maxabs": max(r["partition_residual_maxabs"]
                                             for recs in recs_by_mod.values() for r in recs),
    }
    if mode == "restore_own_update":
        verification["max_swap_residual_maxabs"] = max(c["swap_residual_maxabs"]
                                                       for c in checks_by_mod.values())
    if verify_against_dc_merge and mode == "none":
        import inspect
        accepted = set(inspect.signature(mf.dc_merge).parameters)
        if set(dc_kwargs) <= accepted:
            ref = mf.dc_merge(deltas_dict, **dc_kwargs)
            verification["max_abs_diff_vs_dc_merge"] = max(
                float((ref[k] - merged[k]).abs().max()) for k in merged)
        else:
            verification["dc_merge_comparison_skipped"] = sorted(set(dc_kwargs) - accepted)
    if mode == "control_matched_lane":
        tgt = sum(c.get("target_energy", 0.0) for c in checks_by_mod.values())
        got = sum(c.get("matched_energy", 0.0) for c in checks_by_mod.values())
        verification["control_energy_ratio"] = got / tgt if tgt > 0 else None
        verification["control_underpowered_modules"] = sum(
            bool(c.get("control_underpowered")) for c in checks_by_mod.values())

    summary = summarise(recs_by_mod, task_names)
    tel = max(abs(v["telescoping_residual"]) for v in summary.values())
    verification["max_telescoping_residual"] = tel
    failures = []
    if verification["max_ties_recompute_maxabs"] > atol:
        failures.append("TIES recomputation differs from repository aggregate")
    if verification["max_merge_recompute_maxabs"] > atol:
        failures.append("merged matrix recomputation differs from repository output")
    if verification["total_outside_lane_changed_entries"] != 0:
        failures.append("intervention changed entries outside the edited lane")
    if verification["max_partition_residual_maxabs"] > atol:
        failures.append("TIES class partition does not sum to the lane change")
    if verification.get("max_swap_residual_maxabs", 0.0) > atol:
        failures.append("restore_own_update changed more than the lane-p swap")
    if verification.get("max_abs_diff_vs_dc_merge", 0.0) > atol:
        failures.append("merge differs from dc_merge output")
    if tel > 1e-9:
        failures.append("stage increments do not telescope")
    verification["passed"] = not failures
    verification["failures"] = failures

    result = {"mode": mode, "target": target, "alpha": alpha, "seed": seed,
              "dc_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dc_kwargs.items()},
              "tasks": list(task_names), "verification": verification, "per_task": summary}
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = mode if target is None else f"{mode}__{target}"
        with open(log_dir / f"{stem}.jsonl", "w") as fh:
            for key in recs_by_mod:
                fh.write(json.dumps({"module": key, "checks": checks_by_mod[key],
                                     "records": recs_by_mod[key]}) + "\n")
        (log_dir / f"{stem}.summary.json").write_text(json.dumps(result, indent=2))
    if failures:
        raise AssertionError("consistency check failed: " + "; ".join(failures))
    return merged, result


# ----------------------------------------------------------------------------
# accuracy-level experiment for one context
# ----------------------------------------------------------------------------
def run_context(deltas_dict, solo_deltas, mf, task_names, *, alpha, evaluate: Callable,
                out_dir, dc_kwargs=None, seed=0, solo_accuracy: Mapping[str, float] | None = None,
                modes=("protect_lane", "fix_conflict", "block_injection", "undilute",
                       "control_matched_lane", "restore_own_update")):
    """Direct causal check for one merge context.

    ``deltas_dict``: module -> [delta per task] for this context.
    ``solo_deltas[name]``: module -> [delta] for that task alone (same operator).
    ``evaluate(merged_deltas, alpha) -> {task_name: accuracy}`` wraps the
    pipeline's ``apply_merge`` + ``evaluate_split``.

    Harm h_i = acc_i(solo merge) - acc_i(context merge).  For each intervention
    on target p the log records the change in h_p (benefit) and in every other
    task's harm (cost), next to the weight-space accounting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dc_kwargs = dict(dc_kwargs or {})
    ranks = dc_kwargs.get("task_ranks")

    solo_acc = dict(solo_accuracy or {})
    for i, name in enumerate(task_names):
        if name in solo_acc:          # reuse: a task's solo merge is context-independent
            continue
        kw = dict(dc_kwargs)
        if ranks is not None:
            kw["task_ranks"] = (ranks[i],)
        merged_solo, _ = merge_with_attribution(solo_deltas[name], mf, [name], alpha=alpha,
                                     dc_kwargs=kw, log_dir=out_dir / "solo")
        solo_acc[name] = evaluate(merged_solo, alpha)[name]

    merged0, base0 = merge_with_attribution(deltas_dict, mf, task_names, alpha=alpha,
                                  dc_kwargs=dc_kwargs, log_dir=out_dir)
    acc0 = evaluate(merged0, alpha)
    harm0 = {t: solo_acc[t] - acc0[t] for t in task_names}

    rows = []
    for p in task_names:
        for mode in modes:
            m, aud = merge_with_attribution(deltas_dict, mf, task_names, alpha=alpha, mode=mode,
                                 target=p, seed=seed, dc_kwargs=dc_kwargs, log_dir=out_dir)
            acc = evaluate(m, alpha)
            harm = {t: solo_acc[t] - acc[t] for t in task_names}
            rows.append({
                "target": p, "mode": mode,
                "harm_baseline_target": harm0[p],
                "harm_after_target": harm[p],
                "benefit_target_pp": harm0[p] - harm[p],
                "share_of_target_harm_removed": ((harm0[p] - harm[p]) / harm0[p]
                                                 if harm0[p] > 0 else None),
                "cost_to_others_pp": {t: harm[t] - harm0[t] for t in task_names if t != p},
                "mean_change_pp": sum(acc[t] - acc0[t] for t in task_names) / len(task_names),
                "verification": aud["verification"],
                "weight_space_target": aud["per_task"][p],
            })
    report = {"tasks": list(task_names), "alpha": alpha, "solo_accuracy": solo_acc,
              "context_accuracy": acc0, "harm": harm0,
              "weight_space_baseline": base0["per_task"],
              "baseline_verification": base0["verification"], "interventions": rows}
    (out_dir / "context_report.json").write_text(json.dumps(report, indent=2))
    return report
