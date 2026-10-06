"""CLI: python -m task_demand_prediction {preflight,run,simulate}"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from .design import make_design, validate_design
from .features import FEATURE_DEFINITIONS
from .runtime import RealEvaluator
from .simulation import SimulationEvaluator, synthetic_features
from .storage import atomic_json, seal, hash_file
from .workflow import freeze_features, rebuild_analysis, register, run_stage

def parser():
    p=argparse.ArgumentParser(description='Can Task Merge Demand Be Predicted Before Evaluation?')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('preflight','run','simulate','analyze'):
        s=sub.add_parser(name)
        s.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[1]))
        s.add_argument('--model-dir'); s.add_argument('--data-dir'); s.add_argument('--adapter-dir'); s.add_argument('--head-dir')
        s.add_argument('--reference-dir'); s.add_argument('--output-dir'); s.add_argument('--device',choices=('cpu','cuda'),default='cuda')
        s.add_argument('--min-free-gib',type=float,default=10)
        if name=='run': s.add_argument('--stage',choices=('validation','final-test','both'),required=True)
        if name=='analyze':
            s.add_argument('--stage',choices=('validation','final_test'),required=True)
            s.add_argument('--analysis-output-dir',required=True)
    return p

def _real_preflight(args,stage):
    from pairwise_rank_density.preflight import resolve_paths,preflight
    if not args.output_dir and not os.environ.get('DCMERGE_OUTPUT_DIR'):
        args.output_dir=str(Path(args.repo_root).resolve()/'analysis_outputs'/'task_demand_prediction')
    paths=resolve_paths(args)
    paths.output.parent.mkdir(parents=True,exist_ok=True)
    identity,operational,references=preflight(paths,stage=stage,device=args.device)
    design=make_design(); identity['design_sha256']=design['design_sha256']
    identity['context_manifest_sha256']=__import__('task_demand_prediction.design',fromlist=['sha256_json']).sha256_json(design['context_manifest'])
    design_file=Path(args.repo_root).resolve()/'task_demand_prediction'/'experiment_design.json'
    manifest_file=Path(args.repo_root).resolve()/'task_demand_prediction'/'context_manifest.json'
    prereg_file=Path(args.repo_root).resolve()/'task_demand_prediction'/'preregistration.json'
    frozen_design=json.loads(design_file.read_text(encoding='utf-8'))
    frozen_manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    prereg=json.loads(prereg_file.read_text(encoding='utf-8'))
    if frozen_design != design or frozen_manifest != design['context_manifest']:
        raise ValueError('repository preregistration artifacts differ from generated design')
    if prereg.get('design_sha256')!=design['design_sha256'] or prereg.get('context_manifest_sha256')!=identity['context_manifest_sha256']:
        raise ValueError('preregistration record does not match frozen design/context manifest')
    identity['preregistration_artifacts_sha256']={p.name:hash_file(p) for p in (design_file,manifest_file,prereg_file)}
    code_root=Path(args.repo_root).resolve()
    for package in ('task_demand_prediction','pairwise_rank_density'):
        for p in sorted((code_root/package).glob('*.py')):
            identity['code_sha256'][str(p.relative_to(code_root))]=hash_file(p)
    identity['adapter_checkpoint_sha256']={task:{name:hash_file(paths.adapters/task/name) for name in ('adapter_config.json','adapter_model.bin')}
                                           for task in design['task_order']}
    return paths,identity,operational,references

def main(argv=None):
    args=parser().parse_args(argv); design=make_design(); validate_design(design)
    if args.command=='analyze':
        if not args.output_dir: raise SystemExit('analyze requires --output-dir pointing to sealed experiment outputs')
        result=rebuild_analysis(Path(args.output_dir),args.stage,Path(args.analysis_output_dir))
        print(json.dumps({'status':'analysis regenerated without inference','completion':str(result)},indent=2));return 0
    if args.command=='simulate':
        if not args.output_dir: raise SystemExit('simulate requires --output-dir outside the repository')
        output=Path(args.output_dir).resolve(); repo=Path(args.repo_root).resolve()
        if output==repo or repo in output.parents: raise SystemExit('simulation output must be outside the repository')
        if output.exists() and any(output.iterdir()): raise SystemExit('simulation output must be new or empty')
        identity={'mode':'simulation','design_sha256':design['design_sha256']}
        register(output,identity,'simulation')
        atomic_json(output/'context_manifest.json',seal(design['context_manifest']))
        atomic_json(output/'FEATURES.json',FEATURE_DEFINITIONS)
        freeze_features(output,synthetic_features(design),feature_identity={'synthetic':True},mode='simulation')
        refs={t:80.0 for t in design['task_order']}; evaluator=SimulationEvaluator()
        run_stage(output,'validation',identity,refs,evaluator,mode='simulation')
        run_stage(output,'final_test',identity,refs,evaluator,mode='simulation')
        atomic_json(output/'SIMULATION_ONLY.json',seal({'simulation':True,'scientific_results':False}))
        print(json.dumps({'mode':'simulation','configurations':design['counts'],'output':str(output)},indent=2)); return 0
    stage='final-test' if args.command=='run' and args.stage=='final-test' else 'validation'
    paths,identity,operational,references=_real_preflight(args,stage)
    print(json.dumps({'status':'preflight passed; no inference run','identity':identity,'operational':operational,
                      'design_counts':design['counts'],'protocol_frozen':design['method']['protocol_frozen'],
                      'unresolved_protocol_decisions':design['method']['unresolved_protocol_decisions'],
                      'reusable_oracle_run_count_in_repository':0},indent=2))
    if args.command=='preflight': return 0
    if not design['method'].get('protocol_frozen',False):
        raise SystemExit('real run blocked: protocol is not frozen')
    register(paths.output,identity,'real')
    atomic_json(paths.output/'context_manifest.json',seal(design['context_manifest']))
    definitions=paths.output/'FEATURES.json'
    if definitions.exists():
        if json.loads(definitions.read_text(encoding='utf-8')) != FEATURE_DEFINITIONS:
            raise ValueError('existing feature definitions differ from the registered package; refusing to overwrite')
    else: atomic_json(definitions,FEATURE_DEFINITIONS)
    marker='preflight.validation.DONE.json' if stage=='validation' else 'preflight.final_test.DONE.json'
    atomic_json(paths.output/'status'/marker,seal({'status':'passed','stage':stage,'design_sha256':design['design_sha256'],
                  'identity_sha256':__import__('task_demand_prediction.design',fromlist=['sha256_json']).sha256_json(identity)}))
    evaluator=RealEvaluator(paths,args.device)
    if args.stage in ('validation','both'):
        features=evaluator.build_features()
        freeze_features(paths.output,features,feature_identity={'identity_sha256':__import__('task_demand_prediction.design',fromlist=['sha256_json']).sha256_json(identity)},mode='real')
        run_stage(paths.output,'validation',identity,references,evaluator,mode='real')
    if args.stage in ('final-test','both'):
        # Final-test preflight resolves only the test denominators after validation is sealed.
        paths,identity,operational,references=_real_preflight(args,'final-test')
        atomic_json(paths.output/'status'/'preflight.final_test.DONE.json',seal({'status':'passed','stage':'final-test',
                    'design_sha256':design['design_sha256'],'identity_sha256':__import__('task_demand_prediction.design',fromlist=['sha256_json']).sha256_json(identity)}))
        run_stage(paths.output,'final_test',identity,references,evaluator,mode='real')
    return 0

if __name__=='__main__': raise SystemExit(main())
