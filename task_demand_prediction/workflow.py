"""Sealed registration, resumable stages, grouped validation analyses, and final-test gate."""
from __future__ import annotations
import json, traceback
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .analysis import aggregate_demands, build_oracle_rows, cross_resource_correlations, demand_variation, evaluate_final_test_grouped, evaluate_grouped_splits, feature_correlations
from .design import ACTIVE_RESOURCES, TASKS, make_design, sha256_json, validate_design
from .storage import atomic_csv, atomic_json, atomic_text, hash_file, output_lock, seal, unseal

FEATURES = ("frobenius_norm","squared_frobenius_norm","nuclear_norm","spectral_entropy","effective_rank",
    "energy_rank_25","energy_rank_50","energy_rank_75","energy_rank_90",
    "energy_rank_fraction_25","energy_rank_fraction_50","energy_rank_fraction_75","energy_rank_fraction_90",
    "module_norm_mean","module_norm_std","module_norm_max","module_norm_cv",
    "projection_survival","projection_loss","topk_survival","total_survival",
    "pairwise_cosine_mean","pairwise_cosine_min","pairwise_cosine_max","pairwise_cosine_std",
    "subspace_overlap_mean","unique_direction_score","shared_direction_score","sign_conflict","sign_coverage")

def _config_index(design): return {c['run_id']:c for c in design['configurations']}

def register(output: Path, identity: Mapping[str,Any], mode: str) -> str:
    design=make_design(); validate_design(design)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    context_path=output/'context_manifest.json'; context_value=seal(design['context_manifest'])
    if context_path.exists():
        if unseal(json.loads(context_path.read_text(encoding='utf-8'))) != design['context_manifest']:
            raise ValueError('existing context manifest differs from frozen design')
    else: atomic_json(context_path,context_value)
    payload={"schema_version":1,"design":design,"identity":dict(identity),"mode":mode,
             "context_manifest_sha256":context_value['sha256']}
    path=Path(output)/'registration.json'; value=seal(payload)
    if path.exists():
        old=unseal(json.loads(path.read_text(encoding='utf-8')))
        if old != payload: raise ValueError('existing registration/identity differs; use a new output directory')
    else: atomic_json(path,value)
    return value['sha256']

def freeze_features(output: Path, feature_rows: Sequence[Mapping[str,Any]], *, feature_identity: Mapping[str,Any], mode='real') -> None:
    output=Path(output); design=make_design()
    expected=set()
    for c in design['configurations']:
        if c['focal_task'] is not None: expected.add((c['context_id'],c['focal_task']))
        elif c['phase']=='A': expected.update((c['context_id'],t) for t in c['tasks'])
    got={(r['context_id'],r['focal_task']) for r in feature_rows}
    if got != expected: raise ValueError(f'feature rows incomplete/duplicate (expected {len(expected)}, got {len(got)})')
    path=output/'features.csv'; marker=output/'frozen_features.json'
    if marker.exists():
        old=unseal(json.loads(marker.read_text(encoding='utf-8')))
        if not path.is_file() or hash_file(path) != old['feature_csv_sha256']:
            raise ValueError('frozen feature table is missing or modified')
        if old['mode'] != mode or old['design_sha256'] != design['design_sha256'] or old['feature_identity'] != dict(feature_identity):
            raise ValueError('frozen feature identity differs; refusing to refreeze')
        fd,name=tempfile.mkstemp(prefix='.feature-candidate-',suffix='.csv',dir=output)
        import os
        os.close(fd); candidate=Path(name)
        try:
            atomic_csv(candidate,feature_rows)
            if hash_file(candidate) != old['feature_csv_sha256']:
                raise ValueError('new feature values differ from the frozen feature table')
        finally:
            candidate.unlink(missing_ok=True)
        return
    atomic_csv(path,feature_rows)
    payload={"mode":mode,"design_sha256":design['design_sha256'],"feature_identity":dict(feature_identity),
             "feature_rows":len(feature_rows),"feature_csv_sha256":hash_file(path),"feature_definitions_file":"FEATURES.json",
             "feature_definitions_sha256":hash_file(output/'FEATURES.json') if (output/'FEATURES.json').is_file() else None}
    atomic_json(output/'frozen_features.json',seal(payload))

def _read_csv(path: Path) -> list[dict[str,Any]]:
    import csv
    with Path(path).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def _verify_features(output: Path, design: Mapping[str,Any], mode: str) -> list[dict[str,Any]]:
    marker=output/'frozen_features.json'
    if not marker.is_file(): raise ValueError('feature table and frozen feature manifest are required before evaluation')
    payload=unseal(json.loads(marker.read_text(encoding='utf-8')))
    path=output/'features.csv'
    if payload['mode'] != mode or payload['design_sha256'] != design['design_sha256'] or hash_file(path) != payload['feature_csv_sha256']:
        raise ValueError('feature table does not match its frozen manifest')
    definitions=output/'FEATURES.json'
    if payload.get('feature_definitions_sha256') is not None and hash_file(definitions) != payload['feature_definitions_sha256']:
        raise ValueError('feature definitions changed after freeze')
    rows=_read_csv(path)
    if len(rows) != payload['feature_rows']: raise ValueError('feature row count differs from frozen manifest')
    for row in rows:
        for key in FEATURES:
            if row.get(key) in ('',None): row[key]=None
            else: row[key]=float(row[key])
    return rows

def _write_predictions(output: Path, stage: str, config: Mapping[str,Any], raw: Mapping[str,Any], references: Mapping[str,float], identity_sha: str) -> dict[str,Any]:
    metrics={}
    for task in config['tasks']:
        accuracy=float(raw['per_task'][task]['accuracy_percent']); reference=float(references[task])
        if not 0 <= accuracy <= 100 or reference <= 0: raise ValueError('invalid accuracy or reference')
        metrics[task]={"accuracy_percent":accuracy,"reference_accuracy_percent":reference,
                       "normalized_retention_percent":100*accuracy/reference,"examples":int(raw['per_task'][task]['examples'])}
    result={"mode":raw.get('mode','real'),"run_id":config['run_id'],"split":stage,
            "dataset_split":"val" if stage=='validation' else "test","context_id":config['context_id'],
            "phase":config['phase'],"tasks":config['tasks'],"resource":config['resource'],"focal_task":config['focal_task'],
            "level":config['level'],"scale_by_task":config['scale_by_task'],"rank_by_task":config['rank_by_task'],
            "density_by_task":config['density_by_task'],"metrics":{stage:metrics},
            "density_diagnostics_by_task":raw.get('density_by_task',{}),"resources":raw.get('resources',{}),
            "identity_sha256":identity_sha,"merge_config":{"alpha":.8,"smoothing":"linear","rho":5.0,"top_percent":.001,"aggregation":"ties_small"}}
    path=output/stage/'runs'/config['run_id']/'result.json'; atomic_json(path,result)
    atomic_json(path.parent/'unit.json',seal({"result_sha256":hash_file(path),"identity_sha256":identity_sha}))
    return result

def _load_stage(output: Path, stage: str, design: Mapping[str,Any], identity_sha: str) -> dict[str,Any]:
    result={}
    for config in design['configurations']:
        path=output/stage/'runs'/config['run_id']/'result.json'; marker=path.parent/'unit.json'
        if not marker.exists() or not path.exists(): raise ValueError(f'{stage} incomplete at {config["run_id"]}')
        unit=unseal(json.loads(marker.read_text(encoding='utf-8')))
        if unit != {"result_sha256":hash_file(path),"identity_sha256":identity_sha}: raise ValueError(f'{stage} integrity failure: {config["run_id"]}')
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['run_id'] != config['run_id'] or row['identity_sha256'] != identity_sha: raise ValueError('run metadata mismatch')
        expected={"context_id":config["context_id"],"phase":config["phase"],"tasks":config["tasks"],"resource":config["resource"],
                  "focal_task":config["focal_task"],"level":config["level"],"scale_by_task":config["scale_by_task"],
                  "rank_by_task":config["rank_by_task"],"density_by_task":config["density_by_task"]}
        if any(row.get(k)!=v for k,v in expected.items()): raise ValueError(f'run configuration mismatch: {config["run_id"]}')
        result[config['run_id']]=row
    return result

def _analyze(output: Path, stage: str, design: Mapping[str,Any], results: Mapping[str,Any], feature_rows: Sequence[Mapping[str,Any]], validation_results: Mapping[str,Any]|None=None) -> dict[str,Any]:
    oracle=build_oracle_rows(design,results,stage); demands=aggregate_demands(oracle)
    folder=output/stage
    atomic_csv(folder/'oracle_responses.csv',oracle); atomic_csv(folder/'demand_table.csv',demands)
    density_rows=[]
    config_index=_config_index(design)
    for run_id,result in results.items():
        config=config_index[run_id]
        for task,diag in result.get('density_diagnostics_by_task',{}).items():
            density_rows.append({"split":stage,"run_id":run_id,"context_id":config['context_id'],"phase":config['phase'],
                "focal_task":config['focal_task'],"diagnostic_task":task,"nominal_density":diag.get('nominal_density'),
                "baseline_mask_count":diag.get('baseline_mask_count'),"target_mask_count":diag.get('target_mask_count'),
                "retained_mask_count":diag.get('retained_mask_count'),"retained_nonzero_count":diag.get('retained_nonzero_count'),
                "requested_density":diag.get('requested_density',diag.get('nominal_density')),
                "task_target_mask_count":diag.get('task_target_mask_count',diag.get('target_mask_count')),
                "task_actual_retained_count":diag.get('task_retained_mask_count',diag.get('retained_mask_count')),
                "per_module_counts_json":__import__('json').dumps(diag.get('per_module_counts',{}),sort_keys=True),
                "actual_mask_fraction":diag.get('actual_mask_fraction'),"actual_nonzero_fraction":diag.get('actual_nonzero_fraction'),
                "actual_baseline_support_fraction":diag.get('actual_baseline_support_fraction')})
    atomic_csv(folder/'density_retained_support_audit.csv',density_rows)
    atomic_json(folder/'demand_by_demand_correlations.json',cross_resource_correlations(demands))
    variation=demand_variation(demands); atomic_json(folder/'demand_heterogeneity.json',variation)
    correlations=feature_correlations(feature_rows,demands,FEATURES)
    atomic_csv(folder/'demand_by_feature_correlations.csv',correlations)
    predictive={}
    # Per-task/context feature rows are joined to each separate resource target.
    validation_demands=aggregate_demands(build_oracle_rows(design,validation_results,'validation')) if validation_results is not None else demands
    validation_correlations=feature_correlations(feature_rows,validation_demands,FEATURES)
    for resource in ACTIVE_RESOURCES:
        joined_by_phase={phase:[] for phase in ('A','B','C')}
        validation_joined_by_phase={phase:[] for phase in ('A','B','C')}
        fmap={(r['context_id'],r['focal_task']):r for r in feature_rows}
        for demand in demands:
            if demand['resource'] != resource: continue
            row=fmap[(demand['context_id'],demand['focal_task'])]
            joined_by_phase[row['phase']].append({**row,"target":demand['demand_auc_signed'],"resource":resource})
        predictive[resource]={}
        phase_specs=(('A','focal_task','phase_a_leave_one_task_out'),('B','context_id','phase_b_leave_context_out'),('C','context_id','phase_c_leave_context_out'))
        for phase,group_field,label in phase_specs:
            joined=joined_by_phase[phase]
            validation_joined=[]
            for demand in validation_demands:
                if demand['resource'] != resource: continue
                row=fmap[(demand['context_id'],demand['focal_task'])]
                if row['phase']==phase: validation_joined.append({**row,"target":demand['demand_auc_signed'],"resource":resource})
            if not joined: continue
            if stage == 'validation':
                singles={f:evaluate_grouped_splits(joined,[f],'target',group_field=group_field) for f in FEATURES}
                predictive[resource][label]={"composite_ridge":evaluate_grouped_splits(joined,FEATURES,'target',group_field=group_field),
                    "individual_feature_baselines":singles,
                    "constant_one_baseline":evaluate_grouped_splits(joined,[],'target',group_field=group_field,constant_value=1.0)}
            else:
                single_corr=[r for r in validation_correlations if r['resource']==resource]
                # Preregistered operationalization of the Card's "reasonable"
                # single-feature signal: validation whole-context bootstrap CI
                # excludes zero. Final-test labels are never inspected here.
                signal=any(r.get('bootstrap_ci95') and (r['bootstrap_ci95'][0]>0 or r['bootstrap_ci95'][1]<0) for r in single_corr)
                singles={f:evaluate_final_test_grouped(validation_joined,joined,[f],'target',group_field=group_field) for f in FEATURES}
                composite=evaluate_final_test_grouped(validation_joined,joined,FEATURES,'target',group_field=group_field) if signal else None
                predictive[resource][label]={"composite_ridge":composite,"composite_enabled_by_validation_single_feature_signal":signal,
                    "individual_feature_baselines":singles,
                    "constant_one_baseline":evaluate_final_test_grouped(validation_joined,joined,[],'target',group_field=group_field,constant_value=1.0)}
    atomic_json(folder/'grouped_prediction_results.json',predictive)
    from .plotting import make_plots
    from .report import write_report
    make_plots(output,stage)
    write_report(output,stage,variation,cross_resource_correlations(demands),predictive)
    return {"oracle_rows":len(oracle),"demand_cells":len(demands),"variation":variation,"predictive":predictive}

def run_stage(output: Path, stage: str, identity: Mapping[str,Any], references: Mapping[str,float], evaluator: Callable,
              *, mode='real', feature_rows: Sequence[Mapping[str,Any]]|None=None, force=False) -> None:
    output=Path(output)
    with output_lock(output):
        _run_stage_locked(output,stage,identity,references,evaluator,mode=mode,feature_rows=feature_rows,force=force)

def _run_stage_locked(output: Path, stage: str, identity: Mapping[str,Any], references: Mapping[str,float], evaluator: Callable,
              *, mode='real', feature_rows: Sequence[Mapping[str,Any]]|None=None, force=False) -> None:
    if stage not in ('validation','final_test'): raise ValueError('stage must be validation or final_test')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    design=make_design(); validate_design(design)
    if mode == 'real' and not design['method'].get('protocol_frozen', False):
        raise ValueError('real evaluation blocked: protocol is not frozen')
    reg_sha=register(output,identity,mode); identity_sha=sha256_json(dict(identity))
    features=_verify_features(output,design,mode)
    if stage=='final_test': verify_validation_gate(output,identity_sha,reg_sha,mode)
    status=output/'status'; status.mkdir(parents=True,exist_ok=True)
    atomic_json(status/f'{stage}.RUNNING.json',seal({"mode":mode,"identity_sha256":identity_sha,"registration_sha256":reg_sha,
                "configuration_count":len(design['configurations'])}))
    for config in design['configurations']:
        path=output/stage/'runs'/config['run_id']/'result.json'
        if path.exists() and not force:
            marker=path.parent/'unit.json'
            if marker.exists():
                unit=unseal(json.loads(marker.read_text(encoding='utf-8')))
                if unit.get('result_sha256') != hash_file(path) or unit.get('identity_sha256') != identity_sha: raise ValueError(f'{stage} integrity failure: {config["run_id"]}')
                continue
            raise ValueError(f'incomplete unsealed run exists for {config["run_id"]}; refusing to remove or overwrite it')
        folder=path.parent
        if folder.exists() and any(p.name!='failure.log' for p in folder.iterdir()):
            raise ValueError(f'conflicting partial files exist for {config["run_id"]}; refusing to overwrite them')
        try:
            raw=evaluator(config,'val' if stage=='validation' else 'test')
            result=_write_predictions(output,stage,config,raw,references,identity_sha)
        except Exception:
            folder=output/stage/'runs'/config['run_id']; folder.mkdir(parents=True,exist_ok=True)
            error=traceback.format_exc(); atomic_text(folder/'failure.log',error)
            atomic_json(status/f'{stage}.FAILED.json',seal({"run_id":config['run_id'],"error":error,"identity_sha256":identity_sha}))
            raise
    results=_load_stage(output,stage,design,identity_sha)
    validation_results=_load_stage(output,'validation',design,identity_sha) if stage=='final_test' else None
    report=_analyze(output,stage,design,results,features,validation_results)
    paths=[p for root in (output/stage,output/'figures'/stage) for p in root.glob('**/*')
           if p.is_file() and p.name not in ('COMPLETE.json',)]
    completion={"mode":mode,"identity_sha256":identity_sha,"registration_sha256":reg_sha,
                "design_sha256":design['design_sha256'],"configuration_count":len(design['configurations']),
                "focal_context_rows":96,"analysis":report,"files":{str(p.relative_to(output)):hash_file(p) for p in sorted(paths)}}
    atomic_json(output/stage/'COMPLETE.json',seal(completion))
    complete_hash=hash_file(output/stage/'COMPLETE.json')
    (status/f'{stage}.RUNNING.json').unlink(missing_ok=True)
    atomic_json(status/f'{stage}.DONE.json',seal({"mode":mode,"identity_sha256":identity_sha,"completion_sha256":complete_hash}))
    (status/f'{stage}.FAILED.json').unlink(missing_ok=True)
    if stage=='validation':
        atomic_json(output/'frozen_validation.json',seal({"mode":mode,"identity_sha256":identity_sha,"registration_sha256":reg_sha,
                    "design_sha256":design['design_sha256'],"completion_sha256":complete_hash}))
    else:
        atomic_json(output/'final_test_complete.json',seal({"mode":mode,"identity_sha256":identity_sha,"registration_sha256":reg_sha,
                    "design_sha256":design['design_sha256'],"completion_sha256":complete_hash}))
        atomic_json(status/'ALL_DONE.json',seal({"mode":mode,"identity_sha256":identity_sha,"validation_complete":True,"final_test_complete":True}))

def verify_validation_gate(output: Path, identity_sha: str, registration_sha: str, mode: str) -> None:
    output=Path(output); path=output/'frozen_validation.json'
    if not path.exists(): raise ValueError('final-test blocked: frozen validation manifest missing')
    frozen=unseal(json.loads(path.read_text(encoding='utf-8')))
    if frozen['mode']!=mode or frozen['identity_sha256']!=identity_sha or frozen['registration_sha256']!=registration_sha:
        raise ValueError('final-test blocked: validation identity/registration mismatch')
    complete=output/'validation'/'COMPLETE.json'
    if not complete.exists() or hash_file(complete)!=frozen['completion_sha256']: raise ValueError('final-test blocked: validation completion marker is absent or modified')
    status=output/'status'/'validation.DONE.json'
    if not status.exists(): raise ValueError('final-test blocked: validation status marker is missing')
    status_payload=unseal(json.loads(status.read_text(encoding='utf-8')))
    if status_payload.get('completion_sha256') != frozen['completion_sha256'] or status_payload.get('identity_sha256') != identity_sha:
        raise ValueError('final-test blocked: validation status marker mismatch')
    content=unseal(json.loads(complete.read_text(encoding='utf-8')))
    if content['configuration_count'] != len(make_design()['configurations']): raise ValueError('final-test blocked: validation configuration set incomplete')
    design=make_design(); _load_stage(output,'validation',design,identity_sha)
    required=('oracle_responses.csv','demand_table.csv','demand_by_feature_correlations.csv',
              'demand_by_demand_correlations.json','density_retained_support_audit.csv',
              'demand_heterogeneity.json','grouped_prediction_results.json','REPORT.md')
    if any(not (output/'validation'/name).is_file() for name in required):
        raise ValueError('final-test blocked: required validation analysis outputs are incomplete')
    required_figures=('task_wise_demand_heatmap.png','cross_resource_scatter.png','within_task_context_variation.png',
                      'predicted_vs_oracle.png','leave_one_task_out_results.png')
    if any(not (output/'figures'/'validation'/name).is_file() for name in required_figures):
        raise ValueError('final-test blocked: required validation figures are incomplete')
    for rel,digest in content['files'].items():
        if not (output/rel).is_file() or hash_file(output/rel)!=digest: raise ValueError(f'final-test blocked: validation output modified: {rel}')

def rebuild_analysis(source: Path, stage: str, destination: Path) -> Path:
    """Regenerate analyses/figures from sealed per-run oracle responses, with no evaluator calls."""
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if stage not in ('validation','final_test'): raise ValueError('stage must be validation or final_test')
    if destination==source or source in destination.parents or destination in source.parents:
        raise ValueError('analysis destination must be separate from the source output tree')
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('analysis destination must be new or empty')
    registration=unseal(json.loads((source/'registration.json').read_text(encoding='utf-8')))
    design=registration['design'];identity=registration['identity'];mode=registration['mode'];identity_sha=sha256_json(identity)
    features=_verify_features(source,design,mode)
    results=_load_stage(source,stage,design,identity_sha)
    complete_path=source/stage/'COMPLETE.json'
    complete=unseal(json.loads(complete_path.read_text(encoding='utf-8')))
    if complete['identity_sha256']!=identity_sha: raise ValueError('source stage identity mismatch')
    for rel,digest in complete['files'].items():
        if not (source/rel).is_file() or hash_file(source/rel)!=digest: raise ValueError(f'source artifact integrity failure: {rel}')
    destination.mkdir(parents=True,exist_ok=True)
    atomic_text(destination/'features.csv',(source/'features.csv').read_text(encoding='utf-8'))
    atomic_text(destination/'FEATURES.json',(source/'FEATURES.json').read_text(encoding='utf-8'))
    validation_results=_load_stage(source,'validation',design,identity_sha) if stage=='final_test' else None
    summary=_analyze(destination,stage,design,results,features,validation_results)
    outputs=[p for p in (destination/stage,destination/'figures'/stage) for p in p.glob('**/*') if p.is_file()]
    atomic_json(destination/'REANALYSIS_COMPLETE.json',seal({"source_output":str(source),"source_stage":stage,
        "source_registration_sha256":hash_file(source/'registration.json'),"source_stage_complete_sha256":hash_file(complete_path),
        "feature_csv_sha256":hash_file(source/'features.csv'),"analysis_summary":summary,
        "output_sha256":{str(p.relative_to(destination)):hash_file(p) for p in sorted(outputs)}}))
    return destination/'REANALYSIS_COMPLETE.json'
