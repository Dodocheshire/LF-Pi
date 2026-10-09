"""Human CUB concept audit with a fixed, train-only partial text mapping."""
import json,re,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_cbm.analysis.mechanism import ROOT,write_json,sha

# Same conservative literal rule as PS-Shared-Pi/ps_pi/native_attributes.py;
# copied here to keep the closed PS experiment untouched and independently runnable.
PARTS={'wing':['wing','wings'],'back':['back'],'belly':['belly'],'bill':['bill','beak'],
 'breast':['breast','chest'],'crown':['crown','cap'],'eye':['eye','eyes'],
 'forehead':['forehead'],'leg':['leg','legs'],'nape':['nape'],'throat':['throat'],
 'under_tail':['undertail','under tail'],'upper_tail':['uppertail','upper tail'],
 'underparts':['underparts'],'upperparts':['upperparts'],'primary':['plumage','body']}

def mapping(concepts,attributes):
 result={}
 for aid,name in enumerate(attributes):
  kind,color=name.split('::')
  if not kind.endswith('_color'):continue
  part=kind[len('has_'):-len('_color')]
  if part not in PARTS:continue
  colors=['gray','grey'] if color=='grey' else [color];cids=[]
  for cid,concept in enumerate(concepts):
   text=concept.lower().replace('-',' ')
   if re.search(r'\b(and|bars?|stripes?|streak\w*|spott\w*|patch\w*|tips?|edges?|edged|bands?)\b',text):continue
   if any(re.search(r'\b'+re.escape(c)+r'\s+'+re.escape(p)+r'\b',text) for c in colors for p in PARTS[part]):cids.append(cid)
  if cids:result[aid]=cids
 return result

def threshold(x,y,known):
 x=x[known];y=y[known];order=np.argsort(-x,kind='stable');x=x[order];y=y[order]
 ends=np.flatnonzero(np.r_[x[:-1]!=x[1:],True]);tp=np.cumsum(y)[ends];fp=np.cumsum(~y)[ends]
 ba=.5*(tp/y.sum()+1-fp/(~y).sum());candidates=np.r_[np.nextafter(x[0],np.array(np.inf,dtype=x.dtype)),x[ends]]
 return float(candidates[np.argmax(np.r_[.5,ba])])

def paired_interval(values,replicates=5000):
 values=np.asarray(values,dtype=float);values=values[np.isfinite(values)];rng=np.random.default_rng(20261009)
 means=np.array([values[rng.integers(len(values),size=len(values))].mean() for _ in range(replicates)])
 return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n':len(values),'method':'paired image bootstrap, 5000 replicates; conditional on old checkpoint'}

def main():
 torch.set_num_threads(8);directory=ROOT/'artifacts/mechanism_v1/cub';annotations=ROOT.parent/'D-Pi/artifacts/annotations'
 dataset=torch.load(directory/'dataset.pt',weights_only=True);train=torch.load(directory/'training_reference.pt',weights_only=True)
 test=torch.load(directory/'predictions.pt',weights_only=True);names=[line.split(' ',1)[1] for line in (annotations/'attributes.txt').read_text().splitlines()]
 ids={line.split()[1]:int(line.split()[0]) for line in (annotations/'CUB_200_2011/images.txt').read_text().splitlines()}
 train_ids=[ids['/'.join(Path(dataset['sources']['train'][int(i)][0]).parts[-2:])] for i in train['indices']]
 test_ids=[ids['/'.join(Path(dataset['sources']['test'][int(i)][0]).parts[-2:])] for i in test['indices']]
 assert not set(train_ids)&set(test_ids)
 lookup={iid:(s,j) for s,rows in [('train',train_ids),('test',test_ids)] for j,iid in enumerate(rows)}
 positive={s:np.zeros((len(rows),len(names)),dtype=bool) for s,rows in [('train',train_ids),('test',test_ids)]}
 certainty={s:np.zeros_like(v,dtype=np.uint8) for s,v in positive.items()}
 with (annotations/'CUB_200_2011/attributes/image_attribute_labels.txt').open() as stream:
  for line in stream:
   fields=line.split();iid=int(fields[0])
   if iid in lookup:
    s,j=lookup[iid];aid=int(fields[1])-1;positive[s][j,aid]=bool(int(fields[2]));certainty[s][j,aid]=int(fields[3])
 assignments=mapping(train['names'],names);scores=train['concepts'].sigmoid().numpy();cohort={}
 for aid,cids in assignments.items():
  y=positive['train'][:,aid];known=certainty['train'][:,aid]>=3
  if (known&y).sum()<10 or (known&~y).sum()<10:continue
  cohort[str(aid)]={'name':names[aid],'concept_ids':cids,'concepts':[train['names'][c] for c in cids],
                    'threshold':threshold(scores[:,cids].max(1),y,known),'train_known_positive':int((known&y).sum()),'train_known_negative':int((known&~y).sum())}
 protocol={'source_sha256':sha(__file__),'annotation_sha256':sha(annotations/'CUB_200_2011/attributes/image_attribute_labels.txt'),
           'names_sha256':sha(annotations/'attributes.txt'),'selection':'literal single-color body parts, no patterns; >=10 known train positives and negatives',
           'score':'max sigmoid(standardized LF concept value) over literal matches; MC uses sigmoid of average concept value, not average sigmoid',
           'certainty_minimum':3,'threshold_fit':'4990 historical training images, original clean model only; same threshold for every case',
           'mapped_attributes':len(assignments),'evaluated_attributes':len(cohort),'total_native_attributes':len(names),'cohort':cohort,
           'limits':'Partial whole-image attribute mapping; no localization ground truth for these scores; no new independent model training.'}
 write_json(protocol,directory/'native_protocol.json');rows=[]
 for key,pack in test['predictions'].items():
  probs=pack['concepts'].sigmoid().numpy();per_attribute=[]
  for aid_text,item in cohort.items():
   aid=int(aid_text);known=certainty['test'][:,aid]>=3;y=positive['test'][:,aid][known];score=probs[:,item['concept_ids']].max(1)[known];pred=score>=item['threshold']
   tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
   precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else None
   ba=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None
   auc=float(roc_auc_score(y,score)) if y.any() and not y.all() else None
   per_attribute.append({'aid':aid,'known':len(y),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'ba':ba,'auc':auc})
  macro={metric:float(np.mean([r[metric] for r in per_attribute if r[metric] is not None])) for metric in ['precision','recall','ba','auc']}
  rows.append({'case':key,**{f'macro_{k}':v for k,v in macro.items()},'known_pairs':sum(v['known'] for v in per_attribute),'per_attribute':per_attribute})
 write_json({'rows':rows,'protocol':protocol},directory/'native_summary.json')
 pred=test['predictions'];labels=test['labels'];intervals={}
 for site in ['target_image','target_internal','target_global']:
  for condition in ['original','swap_bg','neutral_bg']:
   a=pred[f'{site}/sample/{condition}']['probabilities'].argmax(1).eq(labels).numpy().astype(float)
   b=pred[f'{site}/mean/{condition}']['probabilities'].argmax(1).eq(labels).numpy().astype(float)
   intervals[f'{site}/sample_minus_mean/{condition}']=paired_interval(100*(a-b))
  a=pred[f'{site}/mean/swap_bg']['probabilities'].argmax(1).eq(labels).numpy().astype(float)
  b=pred['baseline/clean/swap_bg']['probabilities'].argmax(1).eq(labels).numpy().astype(float)
  intervals[f'{site}/mean_minus_baseline/swap_bg']=paired_interval(100*(a-b))
 spatial=json.loads((directory/'target_image_distribution.json').read_text())['spatial']
 intervals['image_sigma_background_minus_foreground']=paired_interval([v['sigma_bg']-v['sigma_fg'] for v in spatial])
 gallery=torch.load(directory/'gallery.pt',weights_only=True)
 # Full diagnostic CAM focus was saved only as the aggregate for adapted mean;
 # do not infer a paired localization CI from the six gallery examples.
 write_json(intervals,directory/'paired_intervals.json')
 print('Native mapped/evaluated:',len(assignments),len(cohort),'of',len(names))
 for r in rows:
  if r['case'].endswith('/original') or r['case'].endswith('/swap_bg'):print({k:v for k,v in r.items() if k!='per_attribute'})
 print('sigma BG-FG',intervals['image_sigma_background_minus_foreground'])

if __name__=='__main__':main()
