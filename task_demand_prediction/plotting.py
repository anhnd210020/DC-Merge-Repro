"""Generate the registered descriptive plots after a stage is complete."""
from __future__ import annotations
import csv, json
import os, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np

def _csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def _savefig(fig,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.stem+'-',suffix=path.suffix,dir=path.parent);os.close(fd)
    try: fig.savefig(name,dpi=160,format=path.suffix.lstrip('.'));os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def make_plots(output: Path, split='validation') -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .design import TASKS
    root=Path(output); folder=root/split; out=root/'figures'/split; out.mkdir(parents=True,exist_ok=True)
    demand=_csv(folder/'demand_table.csv'); responses=_csv(folder/'oracle_responses.csv')
    written=[]
    # Task-by-context demand heatmap, retaining a separate panel per resource type.
    contexts=sorted({r['context_id'] for r in demand})
    resources=tuple(dict.fromkeys(r['resource'] for r in demand))
    fig,axes=plt.subplots(len(resources),1,figsize=(max(12,len(contexts)*.28),max(4,len(resources)*3.3)),constrained_layout=True,squeeze=False)
    for ax,resource in zip(axes[:,0],resources):
        grid=np.full((len(TASKS),len(contexts)),np.nan)
        for r in demand:
            if r['resource']==resource:grid[TASKS.index(r['focal_task']),contexts.index(r['context_id'])]=float(r['demand_auc_signed'])
        im=ax.imshow(grid,aspect='auto',interpolation='nearest',cmap='coolwarm')
        ax.set_yticks(range(len(TASKS)),TASKS); ax.set_title(f'{resource} signed demand by task and context'); ax.set_xticks([])
        fig.colorbar(im,ax=ax,label='signed AUC (pp × resource-axis unit)')
    p=out/'task_wise_demand_heatmap.png'; _savefig(fig,p);plt.close(fig);written.append(str(p))
    # Cross-resource scatter panels.
    by={(r['context_id'],r['focal_task'],r['resource']):float(r['demand_auc_signed']) for r in demand}
    pairs=[(a,b) for i,a in enumerate(resources) for b in resources[i+1:]]
    fig,axes=plt.subplots(1,max(1,len(pairs)),figsize=(max(5,5*len(pairs)),4),constrained_layout=True,squeeze=False)
    for ax,(a,b) in zip(axes[0],pairs):
        xy=[(v,by[(c,t,b)]) for (c,t,res),v in by.items() if res==a and (c,t,b) in by]
        if xy: ax.scatter(*zip(*xy),s=15,alpha=.6)
        ax.set(xlabel=f'{a} demand',ylabel=f'{b} demand',title=f'{a} vs {b}')
    p=out/'cross_resource_scatter.png';_savefig(fig,p);plt.close(fig);written.append(str(p))
    # Within-task context variation, without pooling resource axes.
    fig,axes=plt.subplots(1,len(resources),figsize=(max(5,5*len(resources)),5),constrained_layout=True,squeeze=False)
    for ax,resource in zip(axes[0],resources):
        task_values=defaultdict(list)
        for r in demand:
            if r['resource']==resource:task_values[r['focal_task']].append(float(r['demand_auc_signed']))
        data=[task_values[t] for t in TASKS]; ax.boxplot(data,labels=[t.replace('stanford_','') for t in TASKS],showfliers=False)
        ax.axhline(0,color='black',linewidth=.7);ax.tick_params(axis='x',rotation=70);ax.set_title(resource);ax.set_ylabel('signed demand AUC')
    p=out/'within_task_context_variation.png';_savefig(fig,p);plt.close(fig);written.append(str(p))
    # Predicted versus oracle and per-fold grouped results.
    predictive=json.loads((folder/'grouped_prediction_results.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,len(predictive),figsize=(max(5,5*len(predictive)),4),constrained_layout=True,squeeze=False)
    for ax,(resource,summary) in zip(axes[0],predictive.items()):
        composite=summary.get('phase_a_leave_one_task_out',{}).get('composite_ridge')
        if composite is not None:
            data=composite.get('aggregate_oof') or composite.get('aggregate_final_test')
            target=composite['oof_actual'];pred=composite.get('oof_ridge_prediction',composite.get('oof_prediction',[]))
            ax.scatter(target,pred,s=14,alpha=.6)
            rho=data.get('spearman') if data else None
        else: rho=None
        ax.set_title(f'{resource}: LOTO Spearman={rho}');ax.set_xlabel('oracle demand');ax.set_ylabel('predicted demand')
    p=out/'predicted_vs_oracle.png';_savefig(fig,p);plt.close(fig);written.append(str(p))
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    labels=[];vals=[]
    for resource,summary in predictive.items():
        composite=summary.get('phase_a_leave_one_task_out',{}).get('composite_ridge')
        for fold in (composite or {}).get('folds',[]):
            labels.append(f'{resource}\n{fold["held_out"]}');vals.append(np.nan if fold['spearman'] is None else fold['spearman'])
    ax.bar(range(len(vals)),vals);ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(len(vals)),labels,rotation=75,ha='right');ax.set_ylabel('held-out Spearman');ax.set_title('Leave-one-task-out results')
    p=out/'leave_one_task_out_results.png';_savefig(fig,p);plt.close(fig);written.append(str(p))
    return written

