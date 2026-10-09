"""One fixed-budget matched global-site experiment; no tuning or test selection."""
import copy,json,random,sys
from pathlib import Path
import clip
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_cifar_noise_mechanism import Images,make_cbm,BASE,RUNS
from pi_cbm.analysis.mechanism import ROOT,write_json,sha,gpu_lease,classification,nec
from pi_cbm.generators import NoiseGenerator
from pi_cbm.data import make_splits
from pi_cbm.training.lfcbm import cos_cubed,fit_saga_head

def reset(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def parameters(gen,x,mode):
 if mode=='projection_only':return torch.zeros_like(x),torch.zeros_like(x)
 mu,sigma=gen.distribution(x[:,None,:]);mu=mu[:,0];sigma=sigma[:,0]
 if mode=='mean_only':sigma=torch.zeros_like(sigma)
 return mu,sigma

def forward(cbm,gen,x,mode,random_generator=None,sample=True):
 mu,sigma=parameters(gen,x,mode)
 draw=torch.randn(x.shape,device=x.device,generator=random_generator) if mode=='stochastic' and sample else 0
 c=cbm.concepts(x+mu+sigma*draw)
 return cbm.head(c),c,((mu/gen.feature_scale).square()+(sigma/gen.feature_scale).square()).mean()

@torch.no_grad()
def predict(cbm,gen,x,mode,seed,samples=1):
 rng=torch.Generator(device='cuda').manual_seed(seed);ps=[];cs=[]
 for batch in x.split(256):
  probs=[];concepts=[]
  for j in range(samples):
   logits,c,e=forward(cbm,gen,batch,mode,rng);probs.append(logits.softmax(-1));concepts.append(c)
  ps.append(torch.stack(probs).mean(0).cpu());cs.append(torch.stack(concepts).mean(0).cpu())
 return torch.cat(ps),torch.cat(cs)

def main():
 torch.set_num_threads(8);root=ROOT/'artifacts/mechanism_v1/cifar10';out=root/'matched_global_controls';out.mkdir(parents=True,exist_ok=True)
 original=torch.load(ROOT/'artifacts/cifar-nec-runs'/RUNS['cifar10']['global']/'checkpoint.pt',weights_only=True,map_location='cpu')
 config=original['config'];basepack=torch.load(ROOT/'artifacts/cifar-nec-baselines'/BASE['cifar10'],weights_only=True,map_location='cpu')
 references={s:torch.load(root/f'clean_{s}.pt',weights_only=True,map_location='cpu') for s in ['train','val','test']}
 protocol={'source_sha256':sha(__file__),'baseline_sha256':sha(ROOT/'artifacts/cifar-nec-baselines'/BASE['cifar10']),
           'modes':['projection_only','mean_only','stochastic'],'seed':0,'epochs':20,'budget':'identical batch size 256, projection LR 1e-5, generator LR 3e-4, CE + anchor MSE + teacher cos_cubed + normalized energy; frozen backbone/head during adaptation',
           'initialization':'same original clean projection/head and identical freshly initialized mean/scale generator state; stochastic/control have identical train order',
           'selection':'adapter-val NLL before sparse head refit, common fixed MC1 draw for stochastic; never test',
           'head_refit':'identical SAGA settings; train/val concepts frozen once; separate identical head RNG seed 2000 for every mode',
           'limits':'Single new adaptation seed on one inherited old baseline with test-selected projection/bank; mechanistic pilot, not independent held-out SOTA result.',
           'train_n':45000,'val_n':5000,'test_n':10000}
 guard=out/'protocol.json'
 if guard.exists() and json.loads(guard.read_text())!=protocol:raise RuntimeError('Use a new version for changed protocol')
 write_json(protocol,guard)
 with gpu_lease('LF matched global controls'):
  train_source,_,idx=make_splits(config);text=original['generator']['text'];teacher_values={}
  missing=[s for s in ['train','val'] if not (out/f'teacher_{s}.pt').exists()]
  if missing:
   teacher,_=clip.load('ViT-B/16',device='cuda',jit=False);teacher.eval().requires_grad_(False)
   with torch.no_grad():
    for s in missing:
     chunks=[];ids=[]
     for x,y,i in DataLoader(Images(train_source,references[s]['indices']),batch_size=128,num_workers=4):
      visual=F.normalize(teacher.encode_image(x.cuda()).float().cpu(),dim=-1);chunks.append(visual@text.T);ids.append(i)
     assert torch.equal(torch.cat(ids),references[s]['indices'])
     torch.save(torch.cat(chunks),out/f'teacher_{s}.pt')
   del teacher;torch.cuda.empty_cache()
  xs={s:v['features'].cuda() for s,v in references.items()};ys={s:v['labels'].cuda() for s,v in references.items()}
  teachers={s:torch.load(out/f'teacher_{s}.pt',weights_only=True,map_location='cpu').cuda() for s in ['train','val']}
  base=make_cbm(basepack['cbm']);anchors={s:base.concepts(xs[s]).detach() for s in ['train','val']}
  reset(0);initial=NoiseGenerator(xs['train'].shape[1],text,original['generator']['feature_scale'],config['noise']).cuda();initial_state=copy.deepcopy(initial.state_dict());torch.save({k:v.cpu() for k,v in initial_state.items()},out/'initial_generator.pt')
  rows=[];saved_predictions={}
  for mode in protocol['modes']:
   checkpoint=out/f'{mode}.pt'
   if checkpoint.exists():
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu');cbm=make_cbm(saved['cbm']);gen=copy.deepcopy(initial);gen.load_state_dict(saved['generator']);history=saved['history'];best_epoch=saved['best_epoch']
   else:
    reset(0);cbm=make_cbm(basepack['cbm']);cbm.projection.requires_grad_(True);gen=copy.deepcopy(initial);gen.load_state_dict(initial_state)
    if mode=='projection_only':gen.requires_grad_(False)
    elif mode=='mean_only':gen.scale_head.requires_grad_(False)
    groups=[{'params':list(cbm.projection.parameters()),'lr':1e-5}]
    if mode!='projection_only':groups.append({'params':[p for p in gen.parameters() if p.requires_grad],'lr':3e-4})
    optimizer=torch.optim.Adam(groups);params=[p for group in groups for p in group['params']]
    order_rng=torch.Generator().manual_seed(0);noise_rng=torch.Generator(device='cuda').manual_seed(0);best=float('inf');history=[]
    for epoch in range(20):
     order=torch.randperm(len(xs['train']),generator=order_rng);loss_sum=0.
     for ids in order.split(256):
      logits,c,energy=forward(cbm,gen,xs['train'][ids],mode,noise_rng);loss=F.cross_entropy(logits,ys['train'][ids])+F.mse_loss(c,anchors['train'][ids])+.001*energy+(1-cos_cubed(teachers['train'][ids],c).mean())
      optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();loss_sum+=float(loss.detach())*len(ids)
     val_p,_=predict(cbm,gen,xs['val'],mode,1000);metrics=classification(val_p,references['val']['labels']);history.append({'epoch':epoch+1,'train_loss':loss_sum/45000,'val':metrics})
     print(mode,epoch+1,metrics,flush=True)
     if metrics['nll']<best:best=metrics['nll'];best_epoch=epoch+1;best_cbm=copy.deepcopy(cbm.state_dict());best_gen=copy.deepcopy(gen.state_dict())
    cbm.load_state_dict(best_cbm);gen.load_state_dict(best_gen);cbm.requires_grad_(False);gen.requires_grad_(False)
    _,train_c=predict(cbm,gen,xs['train'],mode,2000);_,val_c=predict(cbm,gen,xs['val'],mode,2001)
    reset(2000);head_metrics=fit_saga_head(cbm.head,train_c,references['train']['labels'],val_c,references['val']['labels'],config['baseline']['head'])
    torch.save({'cbm':{k:v.cpu() for k,v in cbm.state_dict().items()},'generator':{k:v.cpu() for k,v in gen.state_dict().items()},'history':history,'best_epoch':best_epoch,'head_metrics':head_metrics,'protocol':protocol},checkpoint)
   for infer in ([mode,'stochastic_mean'] if mode=='stochastic' else [mode]):
    p,c=predict(cbm,gen,xs['test'],'mean_only' if infer=='stochastic_mean' else mode,20261009,samples=8 if infer=='stochastic' else 1)
    row={'case':infer,**classification(p,references['test']['labels']),'nec':nec(cbm),'best_epoch':best_epoch};rows.append(row);saved_predictions[infer]=p;print('FINAL',row,flush=True)
   write_json({'rows':rows,'protocol':protocol},out/'summary.json')
  torch.save({'probabilities':saved_predictions,'labels':references['test']['labels']},out/'predictions.pt');write_json({'phase':'complete'},out/'status.json')

if __name__=='__main__':main()
