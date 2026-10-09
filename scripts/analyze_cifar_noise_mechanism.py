"""Read-only CIFAR counterfactuals and a clean, matched sparse-head refit."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_cbm.analysis.mechanism import ROOT,sha,write_json,gpu_lease,counterfactual,classification,nec,normalized_energy
from pi_cbm.data import make_splits,digest
from pi_cbm.evaluation.checkpoint import restore_model
from pi_cbm.models.cbm import ConceptBottleneck
from pi_cbm.models.backbones import load_backbone,target_preprocess
from pi_cbm.training.lfcbm import fit_saga_head

RUNS={
 'cifar10':{'image':'20260926-095519-966822-cifar10_target_image_finetune_clip1_lr3e4',
            'internal':'20260926-093530-372710-cifar10_target_internal_finetune_clip1_lr3e4',
            'global':'20260926-093431-788692-cifar10_target_global_finetune_clip1_lr3e4'},
 'cifar100':{'image':'20260926-092202-535249-cifar100_target_image_finetune_clip1_lr3e4',
             'internal':'20260926-092306-797530-cifar100_target_internal_finetune_clip1_lr3e4',
             'global':'20260926-092202-642765-cifar100_target_global_finetune_clip1_lr3e4'}}
BASE={'cifar10':'baseline-9a28e1c2c5321a8d.pt','cifar100':'baseline-0a580a801126afe1.pt'}

class Images(Dataset):
 def __init__(self,source,indices):self.source=source;self.indices=list(map(int,indices));self.transform=target_preprocess('clip_RN50')
 def __len__(self):return len(self.indices)
 def __getitem__(self,j):
  x,y=self.source[self.indices[j]];return self.transform(x),y,self.indices[j]

def make_cbm(state):
 m=ConceptBottleneck(state['projection.weight'].shape[1],state['projection.weight'].shape[0],state['head.weight'].shape[0]);m.load_state_dict(state)
 return m.cuda().eval().requires_grad_(False)

def extract(backbone,loader):
 xs=[];ys=[];ids=[]
 with torch.no_grad():
  for x,y,i in loader:xs.append(backbone(x.cuda()).cpu());ys.append(y);ids.append(i)
 return {'features':torch.cat(xs),'labels':torch.cat(ys),'indices':torch.cat(ids)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(RUNS),required=True);p.add_argument('--samples',type=int,default=8);args=p.parse_args()
 torch.set_num_threads(8);directory=ROOT/'artifacts/mechanism_v1'/args.dataset;directory.mkdir(parents=True,exist_ok=True)
 packs={site:torch.load(ROOT/'artifacts/cifar-nec-runs'/run/'checkpoint.pt',weights_only=True,map_location='cpu') for site,run in RUNS[args.dataset].items()}
 config=packs['global']['config'];train,test,indices=make_splits(config);split=digest({k:v.tolist() for k,v in indices.items()})
 for site,run in RUNS[args.dataset].items():
  expected=json.loads((ROOT/'artifacts/cifar-nec-runs'/run/'metrics.json').read_text())['manifest']['split_sha256']
  if split!=expected:raise RuntimeError('Historical CIFAR split changed')
 protocol={'source_sha256':sha(__file__),'helper_sha256':sha(ROOT/'pi_cbm/analysis/mechanism.py'),
           'checkpoint_sha256':{s:sha(ROOT/'artifacts/cifar-nec-runs'/r/'checkpoint.pt') for s,r in RUNS[args.dataset].items()},
           'baseline_sha256':sha(ROOT/'artifacts/cifar-nec-baselines'/BASE[args.dataset]),'historical_split_sha256':split,
           'samples':args.samples,'evaluation':'complete official test set in original index order',
           'clean_head_control':'unchanged clean projection/normalization, same 45000 train / 5000 adapter val, same SAGA settings, seed 0; no test selection',
           'limits':'Old projection and concept bank were selected using test in upstream recipe. New controls cannot undo this; inference ablations do not isolate stochastic training.'}
 guard=directory/'protocol.json'
 if guard.exists() and json.loads(guard.read_text())!=protocol:raise RuntimeError('Use a new protocol version')
 write_json(protocol,guard)
 with gpu_lease('CIFAR mechanism'):
  backbone=load_backbone(config['backbone'],'cuda');basepack=torch.load(ROOT/'artifacts/cifar-nec-baselines'/BASE[args.dataset],weights_only=True,map_location='cpu');base=make_cbm(basepack['cbm'])
  references={}
  for s,source,ids in [('train',train,indices['train']),('val',train,indices['val']),('test',test,np.arange(len(test)))]:
   path=directory/f'clean_{s}.pt'
   if not path.exists():torch.save(extract(backbone,DataLoader(Images(source,ids),batch_size=128,num_workers=4)),path)
   references[s]=torch.load(path,weights_only=True,map_location='cpu')
  clean_scores={s:base.concepts(v['features'].cuda()).cpu() for s,v in references.items()}
  rows=[];predictions={};labels=references['test']['labels']
  p=base.head(clean_scores['test'].cuda()).softmax(-1).detach().cpu();predictions['baseline']=p;rows.append({'site':'baseline','mode':'clean',**classification(p,labels),'nec':nec(base)})
  print(args.dataset,'baseline',rows[-1],flush=True)
  head_path=directory/'clean_refit.pt'
  if head_path.exists():base.load_state_dict(torch.load(head_path,weights_only=True,map_location='cpu')['cbm'])
  else:
   random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
   metrics=fit_saga_head(base.head,clean_scores['train'],references['train']['labels'],clean_scores['val'],references['val']['labels'],config['baseline']['head'])
   torch.save({'cbm':{k:v.cpu() for k,v in base.state_dict().items()},'metrics':metrics,'protocol':protocol},head_path)
  p=base.head(clean_scores['test'].cuda()).softmax(-1).detach().cpu();predictions['clean_refit']=p;rows.append({'site':'clean_refit','mode':'clean',**classification(p,labels),'nec':nec(base)})
  print(args.dataset,'clean_refit',rows[-1],flush=True)
  del clean_scores,references
  loader=DataLoader(Images(test,np.arange(len(test))),batch_size=32,num_workers=4)
  with torch.no_grad():
   for site,pack in packs.items():
    suffix=backbone if site=='image' else (backbone.suffix if site=='internal' else torch.nn.Identity());model=restore_model(pack,'cuda',suffix)
    values={k:[] for k in ['plain','mean','sample']};energy=[];one_draw=[]
    rng=torch.Generator(device='cuda').manual_seed(20261009)
    for x,y,i in loader:
     x=x.cuda();inputs=x if site=='image' else (backbone.prefix(x) if site=='internal' else backbone(x))
     for mode in values:
      p,z,extra=counterfactual(model,inputs,mode,args.samples,rng);values[mode].append(p.cpu())
      if mode=='sample':one_draw.append(extra['sample_probabilities'][0].cpu())
      if mode=='mean':
       m,v=normalized_energy(extra['mean'],extra['scale'],model.generator.feature_scale);energy.extend(zip(m.cpu().tolist(),v.cpu().tolist()))
    values['single_draw']=one_draw
    for mode,batches in values.items():
     p=torch.cat(batches);predictions[f'{site}/{mode}']=p;row={'site':site,'mode':mode,**classification(p,labels),'nec':nec(model.cbm)};rows.append(row);print(args.dataset,site,mode,row,flush=True)
    write_json({'mean_energy':float(np.mean([m for m,v in energy])),'variance_energy':float(np.mean([v for m,v in energy]))},directory/f'{site}_distribution.json')
    write_json({'rows':rows,'protocol':protocol},directory/'summary.json')
  torch.save({'probabilities':predictions,'labels':labels,'indices':torch.arange(len(labels))},directory/'predictions.pt');write_json({'phase':'complete'},directory/'status.json')

if __name__=='__main__':main()
