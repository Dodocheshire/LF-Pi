"""Copy compact evidence tables, hashes and paired intervals into the report."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_cbm.analysis.mechanism import ROOT,sha,write_json
from audit_cub_native_mechanism import paired_interval

OUT=ROOT/'reports/noise_mechanism_v1';DATA=ROOT/'artifacts/mechanism_v1'
def table(rows,path):
 keys=list(dict.fromkeys(k for r in rows for k in r));keys=[k for k in keys if all(not isinstance(r.get(k), (dict,list)) for r in rows)]
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def native_intervals():
 torch.set_num_threads(8);p=DATA/'cub';protocol=json.loads((p/'native_protocol.json').read_text());cohort=protocol['cohort'];d=torch.load(p/'dataset.pt',weights_only=True);v=torch.load(p/'predictions.pt',weights_only=True)
 anno=ROOT.parent/'D-Pi/artifacts/annotations/CUB_200_2011';ids={line.split()[1]:int(line.split()[0]) for line in (anno/'images.txt').read_text().splitlines()}
 image_ids=[ids['/'.join(Path(d['sources']['test'][int(i)][0]).parts[-2:])] for i in v['indices']];lookup={iid:j for j,iid in enumerate(image_ids)};aids=list(map(int,cohort));aid_lookup={aid:j for j,aid in enumerate(aids)}
 y=np.zeros((len(image_ids),len(aids)),dtype=bool);known=np.zeros_like(y)
 with (anno/'attributes/image_attribute_labels.txt').open() as stream:
  for line in stream:
   f=line.split();iid=int(f[0]);aid=int(f[1])-1
   if iid in lookup and aid in aid_lookup:
    j=lookup[iid];k=aid_lookup[aid];y[j,k]=bool(int(f[2]));known[j,k]=int(f[3])>=3
 rng=np.random.default_rng(20261009);weights=rng.multinomial(len(y),np.ones(len(y))/len(y),size=2000).astype(np.float32);values={};point={}
 for key,pack in v['predictions'].items():
  if not key.endswith('/original') or '/plain/' in key:continue
  scores=pack['concepts'].sigmoid().numpy();pred=np.stack([scores[:,cohort[str(a)]['concept_ids']].max(1)>=cohort[str(a)]['threshold'] for a in aids],axis=1)
  tp=(pred&y&known).astype(np.float32);fp=(pred&~y&known).astype(np.float32);btp=weights@tp;bfp=weights@fp
  values[key]=np.divide(btp,btp+bfp,out=np.zeros_like(btp),where=(btp+bfp)>0).mean(1)
  tp0=tp.sum(0);fp0=fp.sum(0);point[key]=float(np.divide(tp0,tp0+fp0,out=np.zeros_like(tp0),where=(tp0+fp0)>0).mean())
 output={}
 for site in ['target_image','target_internal','target_global']:
  for mode in ['mean','sample']:
   key=f'{site}/{mode}/original';base='baseline/clean/original';delta=100*(values[key]-values[base]);output[key+'_minus_baseline_precision_pp']={'mean':100*(point[key]-point[base]),'ci95':np.quantile(delta,[.025,.975]).tolist(),'method':'paired image bootstrap 2000; fixed 48-attribute cohort and train-only thresholds; zero prediction precision=0'}
 write_json(output,OUT/'native_precision_intervals.json')

def main():
 OUT.mkdir(parents=True,exist_ok=True);tables=[];intervals={}
 for name in ['cifar10','cifar100']:
  p=DATA/name;summary=json.loads((p/'summary.json').read_text());table(summary['rows'],OUT/f'{name}_results.csv');write_json(summary['protocol'],OUT/f'{name}_protocol.json');v=torch.load(p/'predictions.pt',weights_only=True);labels=v['labels'];prob=v['probabilities']
  for site in ['image','internal','global']:
   a=prob[f'{site}/sample'].argmax(1).eq(labels).numpy().astype(float);b=prob[f'{site}/mean'].argmax(1).eq(labels).numpy().astype(float);intervals[f'{name}/{site}/sample_minus_mean_pp']=paired_interval(100*(a-b))
  a=prob['clean_refit'].argmax(1).eq(labels).numpy().astype(float);b=prob['baseline'].argmax(1).eq(labels).numpy().astype(float);intervals[f'{name}/clean_refit_minus_baseline_pp']=paired_interval(100*(a-b))
 controls=DATA/'cifar10/matched_global_controls';summary=json.loads((controls/'summary.json').read_text());table(summary['rows'],OUT/'matched_global_controls.csv');write_json(summary['protocol'],OUT/'matched_global_controls_protocol.json');v=torch.load(controls/'predictions.pt',weights_only=True);labels=v['labels']
 for a,b in [('stochastic','mean_only'),('mean_only','projection_only'),('stochastic','stochastic_mean')]:
  ca=v['probabilities'][a].argmax(1).eq(labels).numpy().astype(float);cb=v['probabilities'][b].argmax(1).eq(labels).numpy().astype(float);intervals[f'matched_global/{a}_minus_{b}_pp']=paired_interval(100*(ca-cb))
 write_json(intervals,OUT/'cifar_paired_intervals.json')
 p=DATA/'cub';summary=json.loads((p/'summary.json').read_text());table(summary['rows'],OUT/'cub_results.csv')
 native=json.loads((p/'native_summary.json').read_text());table(native['rows'],OUT/'cub_native_results.csv')
 per_attr=[{'case':r['case'],**a} for r in native['rows'] for a in r['per_attribute']];table(per_attr,OUT/'cub_native_per_attribute.csv')
 for name in ['protocol.json','native_protocol.json','paired_intervals.json']:(OUT/f'cub_{name}').write_bytes((p/name).read_bytes())
 energy=[]
 for name in ['cifar10','cifar100','cub']:
  for file in (DATA/name).glob('*distribution.json'):
   d=json.loads(file.read_text());m=d['mean_energy'];v=d['variance_energy'];energy.append({'dataset':name,'site':file.stem.replace('_distribution',''),'mean_energy':m,'variance_energy':v,'mean_fraction_of_total_energy':m/(m+v)})
 table(energy,OUT/'distribution_energy.csv');native_intervals()
 paths=[ROOT/'pi_cbm/analysis/mechanism.py',*ROOT.glob('scripts/*mechanism*.py'),ROOT/'scripts/train_cifar10_mechanism_controls.py']
 manifest={'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'report_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_manifest.json'},
           'checkpoints':'archived hashes in per-dataset protocol; new matched checkpoints stay under artifacts/mechanism_v1/cifar10/matched_global_controls',
           'scope':'3 datasets, 6 CIFAR archived noisy checkpoints, 3 CUB noisy checkpoints, 2 clean head refits, 3 new matched global control trainings; one adaptation seed'}
 write_json(manifest,OUT/'artifact_manifest.json');print('Compact tables, paired CIs and report manifest exported',flush=True)

if __name__=='__main__':main()
