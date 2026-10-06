"""Adapter/context loader, structural feature preparation, and held-out inference."""
from __future__ import annotations
import os, sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

from .design import ALPHA, RHO, SEED, TASKS, TOP_PERCENT, make_design
from .features import build_feature_row, compute_task_svd_cache, self_structure

def _vision_imports(repo: Path):
    vision=repo/'vision_lora_merge'
    if str(vision) not in sys.path: sys.path.insert(0,str(vision))
    os.chdir(vision)
    from ft_handlers import aggregate_deltas, get_ft_parameters_delta, apply_merge
    from merging_functions import dc_merge
    from utils import get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, evaluate_cliphead
    return vision,aggregate_deltas,get_ft_parameters_delta,apply_merge,dc_merge,get_clip_encodings,get_config_from_name,prepare_experiment_config,set_seed,evaluate_cliphead

class RealEvaluator:
    mode='real'
    def __init__(self, paths, device: str): self.paths=paths; self.device=device; self.runtime=None
    def _load(self):
        if self.runtime is not None: return self.runtime
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled',DCMERGE_MODEL_DIR=str(self.paths.model),DCMERGE_DATA_DIR=str(self.paths.data))
        (vision,aggregate_deltas,get_delta,apply_merge,dc_merge,get_head,get_config,prepare,set_seed,evaluate)=_vision_imports(self.paths.repo)
        import torch
        set_seed(SEED)
        raw=get_config('vitB32_r16_8task_selftrained',device='cpu')
        raw['model']['cachedir']=str(self.paths.model); raw['model']['bases']=[]
        for cfg in raw['dataset']: cfg['clip_encodings']=str(self.paths.heads/'ViT-B-32'/f"{cfg['name']}_head.pt")
        if [cfg['name'] for cfg in raw['dataset']] != list(TASKS): raise RuntimeError('task order mismatch')
        prepared=prepare(raw)
        heads={t:get_head(self.paths.heads/'ViT-B-32'/f'{t}_head.pt').to(self.device) for t in TASKS}
        deltas={}
        for task in TASKS:
            path=self.paths.adapters/task/'adapter_model.bin'
            state=torch.load(path,map_location='cpu',weights_only=True)
            normalized={}
            for key,value in state.items():
                key=str(key)
                if key.startswith('base_model.model.'): key='vision_model.'+key
                key=key.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
                normalized[key]=value
            delta=get_delta(normalized)
            deltas[task]=OrderedDict((key,value.cpu()) for key,value in delta.items())
            del state,normalized,delta
        module_order=tuple(deltas[TASKS[0]])
        if any(tuple(deltas[t]) != module_order for t in TASKS): raise RuntimeError('adapter delta module order mismatch')
        self.runtime={"torch":torch,"aggregate_deltas":aggregate_deltas,"apply_merge":apply_merge,"dc_merge":dc_merge,
                      "evaluate":evaluate,"prepared":prepared,"heads":heads,"deltas":deltas}
        return self.runtime
    def _context_delta(self,tasks):
        r=self._load(); return r['aggregate_deltas']([r['deltas'][t] for t in tasks])
    def build_features(self) -> list[dict[str,Any]]:
        r=self._load(); features=[]; seen=set()
        subspace_cache={t:compute_task_svd_cache(r['deltas'][t]) for t in TASKS}
        self_cache={t:self_structure(t,r['deltas'][t],subspace_cache[t]) for t in TASKS}
        for config in make_design()['configurations']:
            context_id=config['context_id']; tasks=tuple(config['tasks'])
            if context_id in seen: continue
            seen.add(context_id); grouped=self._context_delta(tasks)
            _merged,details=r['dc_merge'](grouped,smoothing_strategy='linear',rho=RHO,return_intermediates=True)
            projected={t:{} for t in tasks}; filtered={t:{} for t in tasks}; preprojection={t:{} for t in tasks}
            for module,diag in details.items():
                for i,t in enumerate(tasks):
                    projected[t][module]=diag['projected_matrices'][i]
                    filtered[t][module]=diag['filtered_matrices'][i]
                    u=diag['singular_directions_u'][i]; s=diag['smoothed_singular_values'][i]; v=diag['singular_directions_v'][i]
                    preprojection[t][module]=u @ __import__('torch').diag(s) @ v
            vectors={t:r['deltas'][t] for t in tasks}
            if config['phase']=='A': focals=list(tasks)
            elif config['phase']=='B': focals=list(dict.fromkeys(row['focal_task'] for row in make_design()['context_manifest']['phase_b_focal_contexts'] if row['context_id']==context_id))
            else: focals=list(next(row['focal_tasks'] for row in make_design()['context_manifest']['phase_c_reused_contexts'] if row['context_id']==context_id))
            for focal in focals:
                features.append(build_feature_row(context_id,config['phase'],focal,list(tasks),vectors,projected=projected,filtered=filtered,preprojection=preprojection,
                                                   subspace_cache=subspace_cache,self_feature_cache=self_cache))
            del grouped,_merged,details
        return features
    def __call__(self,configuration: Mapping[str,Any],split: str) -> dict[str,Any]:
        r=self._load(); torch=r['torch']; tasks=tuple(configuration['tasks']); grouped=self._context_delta(tasks)
        scales=tuple(configuration['scale_by_task'][t] for t in tasks)
        ranks=tuple(configuration['rank_by_task'][t] for t in tasks)
        densities=tuple(configuration['density_by_task'][t] for t in tasks)
        with torch.no_grad():
            merged,details=r['dc_merge'](grouped,smoothing_strategy='linear',rho=RHO,task_scales=scales,task_ranks=ranks,task_densities=densities,return_intermediates=True)
            model=r['apply_merge'](r['prepared']['models']['new'],merged,scaling_coeffs=ALPHA).to(self.device)
        density={t:{'nominal_density':densities[i],'requested_density':densities[i],
                    'baseline_mask_count':0,'target_mask_count':0,'retained_mask_count':0,
                    'retained_nonzero_count':0,'total_coordinates':0,'per_module_counts':{}}
                 for i,t in enumerate(tasks)}
        for module,diag in details.items():
            for i,t in enumerate(tasks):
                stats=diag['density'][i]
                for k in ('baseline_mask_count','target_mask_count','retained_mask_count','retained_nonzero_count','total_coordinates'): density[t][k]+=int(stats[k])
                density[t]['per_module_counts'][str(module)] = {
                    'baseline_mask_count':int(stats['baseline_mask_count']),
                    'target_mask_count':int(stats['target_mask_count']),
                    'retained_mask_count':int(stats['retained_mask_count']),
                }
        per_task={}
        try:
            for t in tasks:
                row=density[t]; row['actual_mask_fraction']=row['retained_mask_count']/row['total_coordinates'] if row['total_coordinates'] else None
                row['actual_nonzero_fraction']=row['retained_nonzero_count']/row['total_coordinates'] if row['total_coordinates'] else None
                row['actual_baseline_support_fraction']=row['retained_mask_count']/row['baseline_mask_count'] if row['baseline_mask_count'] else None
                loader=r['prepared']['data'][TASKS.index(t)]['test'][split]
                acc=r['evaluate'](model,loader,class_vectors=r['heads'][t])
                examples=len(loader.dataset)
                per_task[t]={'accuracy_percent':100*float(acc),'examples':examples}
        finally:
            model.cpu(); del model,merged,details
            if self.device=='cuda': torch.cuda.empty_cache()
        return {'mode':'real','per_task':per_task,'density_by_task':density,'resources':{'device':self.device,'evaluated_tasks':list(tasks)}}
