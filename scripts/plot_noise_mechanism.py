"""Publication artifacts drawn only from real checkpoint diagnostics."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_cbm.analysis.mechanism import ROOT,write_json

OUT=ROOT/'reports/noise_mechanism_v1';DATA=ROOT/'artifacts/mechanism_v1/cub'
COLORS={'baseline':'#64748B','plain':'#A855F7','mean':'#F59E0B','sample':'#0891B2','clean_refit':'#16A34A'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)

def gallery():
 g=torch.load(DATA/'gallery.pt',weights_only=True);std=torch.tensor([.229,.224,.225])[None,:,None,None];mean=torch.tensor([.485,.456,.406])[None,:,None,None]
 clean=g['images']*std+mean;det=(g['images']+g['mean'])*std+mean;sample=g['sample']*std+mean
 mu=(g['mean']*std).square().mean(1).sqrt()*100;sigma=(g['scale']*std).square().mean(1).sqrt()*100
 vmax_mu=float(mu.max());vmax_sigma=float(sigma.max());n=len(clean)
 fig,axes=plt.subplots(n,5,figsize=(13,2.35*n),layout='constrained');titles=['Original','Learned mean: x + mu','Real draw: x + mu + sigma*eps','Mean RMS (% RGB)','Std RMS (% RGB)']
 stats=[]
 for j in range(n):
  label=int(g['labels'][j]);name=f"{label+1:03d}. "
  dataset=torch.load(DATA/'dataset.pt',weights_only=True);path=Path(dataset['sources']['train'][int(g['indices'][j])][0]);name=path.parent.name.replace('_',' ').split('.',1)[1]
  for k,image in enumerate([clean,det,sample]):
   axes[j,k].imshow(image[j].clamp(0,1).permute(1,2,0));p=g[['baseline_probabilities','mean_probabilities','sample_probabilities'][k]][j]
   axes[j,k].set_xlabel(f'p(true)={float(p[label]):.3f}; pred #{int(p.argmax())+1}')
  im_mu=axes[j,3].imshow(mu[j],cmap='magma',vmin=0,vmax=vmax_mu);im_sigma=axes[j,4].imshow(sigma[j],cmap='viridis',vmin=0,vmax=vmax_sigma)
  axes[j,0].set_ylabel(name,fontsize=10)
  for k in range(5):
   axes[j,k].set_xticks([]);axes[j,k].set_yticks([])
   if j==0:axes[j,k].set_title(titles[k],fontsize=11)
  stats.append({'source_index':int(g['indices'][j]),'class':name,'mean_out_of_rgb_fraction':float(((det[j]<0)|(det[j]>1)).float().mean()),'sample_out_of_rgb_fraction':float(((sample[j]<0)|(sample[j]>1)).float().mean()),'mu_rms_max_percent':float(mu[j].max()),'sigma_rms_max_percent':float(sigma[j].max())})
 fig.colorbar(im_mu,ax=axes[:,3],shrink=.35,label='% of RGB [0,1] range');fig.colorbar(im_sigma,ax=axes[:,4],shrink=.35,label='% of RGB [0,1] range')
 fig.suptitle('LF-Pi real image perturbations | fixed validation examples, no success screening\nNo visual amplification. Shared scales within each heatmap column; clipping is display-only.',fontsize=13)
 save(fig,'01_real_image_noise');write_json({'images':stats,'mu_shared_max_percent':vmax_mu,'sigma_shared_max_percent':vmax_sigma},OUT/'gallery_display.json')
 fig,axes=plt.subplots(2,n,figsize=(2.45*n,5.4),layout='constrained')
 for j in range(n):
  cams=[g['baseline_cam'][j],g['mean_cam'][j]];maximum=max(float(c.max()) for c in cams)
  for r,cam in enumerate(cams):
   axes[r,j].imshow(clean[j].clamp(0,1).permute(1,2,0));axes[r,j].imshow(cam/max(maximum,1e-20),cmap='inferno',vmin=0,vmax=1,alpha=.55)
   axes[r,j].contour(g['masks'][j,0],levels=[.5],colors=['#22D3EE'],linewidths=1)
   fraction=float((cam*g['masks'][j,0]).sum()/cam.sum().clamp_min(1e-20));axes[r,j].set_title(f'Positive CAM inside bird: {100*fraction:.1f}%')
   axes[r,j].set_xticks([]);axes[r,j].set_yticks([])
  axes[0,0].set_ylabel('Original LF-CBM');axes[1,0].set_ylabel('LF-Pi image mean')
 fig.suptitle('True-class linear CAM with human foreground outline (cyan)\nCAM normalized by shared original/mean maximum per example. Conditional CAM, not full generator attribution.',fontsize=12)
 save(fig,'02_foreground_cam')

def cub_results():
 rows=json.loads((DATA/'summary.json').read_text())['rows'];native=json.loads((DATA/'native_summary.json').read_text())
 fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');cases=[('baseline','clean'),('target_image','mean'),('target_internal','mean'),('target_global','mean')];labels=['Original LF-CBM','Image mean','Internal mean','Global mean'];colors=['#64748B','#F59E0B','#16A34A','#8B5CF6']
 x=np.arange(4);width=.32
 for j,condition in enumerate(['original','swap_bg']):
  values=[100*next(r['accuracy'] for r in rows if r['site']==s and r['mode']==m and r['condition']==condition) for s,m in cases]
  bars=axes[0].bar(x+(j-.5)*width,values,width,color=colors,alpha=1 if j==0 else .45,hatch=None if j==0 else '//',label=['Original image','Background replaced'][j])
  axes[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
 axes[0].set_xticks(x,labels,rotation=22,ha='right');axes[0].set_ylabel('Task accuracy (%)');axes[0].set_ylim(0,88);axes[0].legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2);axes[0].set_title('Same 800 test images + human masks',pad=35)
 keys=['baseline/clean/original','target_image/mean/original','target_internal/mean/original','target_global/mean/original']
 nr=[next(r for r in native['rows'] if r['case']==key) for key in keys]
 for j,metric in enumerate(['macro_precision','macro_ba']):
  bars=axes[1].bar(x+(j-.5)*width,[100*r[metric] for r in nr],width,color=colors,alpha=1 if j==0 else .45,hatch=None if j==0 else '//',label=['Concept precision','Concept balanced acc.'][j]);axes[1].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
 axes[1].set_xticks(x,labels,rotation=22,ha='right');axes[1].set_ylabel('Human attribute macro score (%)');axes[1].set_ylim(0,100);axes[1].legend(fontsize=8);axes[1].set_title(f"Partial concept audit: {native['protocol']['evaluated_attributes']}/312 attributes\nFixed clean train-only thresholds")
 modes=['plain','mean','sample'];sites=['target_image','target_internal','target_global'];xx=np.arange(3)
 for j,mode in enumerate(modes):
  values=[]
  for site in sites:
   r=next(r for r in rows if r['site']==site and r['mode']==mode and r['condition']=='original');values.append(100*r['accuracy'])
  bars=axes[2].bar(xx+(j-1)*.24,values,.24,color=COLORS[mode],label={'plain':'Adapted CBM; generator off','mean':'Learned mean only','sample':'Mean + random (MC8)'}[mode]);axes[2].bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
 axes[2].set_xticks(xx,['Image','Internal','Global']);axes[2].set_ylim(0,88);axes[2].set_ylabel('Task accuracy (%)');axes[2].legend(fontsize=8);axes[2].set_title('Same adapted checkpoint at inference')
 fig.suptitle('CUB mechanism diagnostics | historical seed-0 models, conditional evidence',fontsize=13);save(fig,'03_cub_mechanism')
 spatial=json.loads((DATA/'target_image_distribution.json').read_text())['spatial'];intervals=json.loads((DATA/'paired_intervals.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for j,key in enumerate(['mean','sigma']):
  values=[100*np.nanmean([s[f'{key}_{region}'] for s in spatial]) for region in ['fg','bg']]
  bars=axes[0].bar(np.arange(2)+(j-.5)*.3,values,.3,color=['#F59E0B','#0891B2'][j],label=['Mean RMS','Standard deviation RMS'][j]);axes[0].bar_label(bars,fmt='%.3f',padding=3)
 axes[0].set_xticks([0,1],['Bird foreground','Background']);axes[0].set_ylabel('% of RGB [0,1] range');axes[0].legend();axes[0].set_title('Region-normalized amplitude, 800 images')
 values=[]
 for site in sites:
  mean=next(r['nll'] for r in rows if r['site']==site and r['mode']=='mean' and r['condition']=='original');sample=next(r['nll'] for r in rows if r['site']==site and r['mode']=='sample' and r['condition']=='original');values.append(sample-mean)
 axes[1].bar(sites,values,color='#0891B2');axes[1].set_xticks(range(3),['Image','Internal','Global']);axes[1].axhline(0,color='#475569',lw=1);axes[1].set_ylabel('MC8 NLL minus mean-only NLL');axes[1].set_title('Random sampling adds almost no change')
 save(fig,'04_noise_amplitude')

def cifar_results():
 dirs=[ROOT/'artifacts/mechanism_v1'/name for name in ['cifar10','cifar100']]
 if not all((d/'summary.json').exists() and (d/'status.json').exists() for d in dirs):return
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for i,(name,directory) in enumerate(zip(['CIFAR-10','CIFAR-100'],dirs)):
  rows=json.loads((directory/'summary.json').read_text())['rows'];cases=[('baseline','clean'),('clean_refit','clean'),('image','mean'),('internal','mean'),('global','mean')]
  labels=['Original','Clean head refit','Image mean','Internal mean','Global mean'];colors=['#64748B','#16A34A','#F59E0B','#0891B2','#8B5CF6'];sel=[next(r for r in rows if r['site']==s and r['mode']==m) for s,m in cases]
  for j,(metric,scale,ylabel) in enumerate([('accuracy',100,'Accuracy (%)'),('nec',1,'NEC: active connections / class')]):
   ax=axes[i,j];bars=ax.bar(labels,[scale*r[metric] for r in sel],color=colors);ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_xticks(range(5),labels,rotation=18,ha='right');ax.set_ylabel(ylabel);ax.set_title(name+' | full 10,000-image test');ax.set_ylim(0,max(scale*r[metric] for r in sel)*1.18)
 fig.suptitle('NEC falls after adaptation + sparse refit; clean refit isolates the optimizer effect\nMean-only inference of historical adapted models; no matched retraining claim',fontsize=13);save(fig,'05_cifar_accuracy_nec')
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for i,(name,directory) in enumerate(zip(['CIFAR-10','CIFAR-100'],dirs)):
  rows=json.loads((directory/'summary.json').read_text())['rows'];x=np.arange(3)
  for j,mode in enumerate(['plain','mean','sample']):
   vals=[100*next(r['accuracy'] for r in rows if r['site']==s and r['mode']==mode) for s in ['image','internal','global']];bars=axes[i].bar(x+(j-1)*.24,vals,.24,color=COLORS[mode],label={'plain':'Generator off','mean':'Mean only','sample':'MC8 random'}[mode]);axes[i].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
  axes[i].set_xticks(x,['Image','Internal','Global']);axes[i].set_ylim(0,105);axes[i].set_ylabel('Test accuracy (%)');axes[i].set_title(name);axes[i].legend(fontsize=8)
 save(fig,'06_cifar_sampling')

def matched_controls():
 path=ROOT/'artifacts/mechanism_v1/cifar10/matched_global_controls/summary.json'
 if not path.exists():return
 rows=json.loads(path.read_text())['rows'];base=json.loads((path.parent.parent/'summary.json').read_text())['rows'][0]
 keys=['projection_only','mean_only','stochastic'];data=[base]+[next(r for r in rows if r['case']==k) for k in keys]
 labels=['Original LF-CBM','Task projection only','Mean adapter','Stochastic adapter'];colors=['#64748B','#16A34A','#F59E0B','#0891B2']
 fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
 for ax,metric,scale,ylabel in zip(axes,['accuracy','nll','nec'],[100,1,1],['Test accuracy (%)','Test NLL (lower is better)','NEC: connections / class']):
  values=[r[metric]*scale for r in data];bars=ax.bar(labels,values,color=colors);ax.bar_label(bars,fmt='%.2f' if metric!='nll' else '%.3f',padding=3);ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.18)
 fig.suptitle('New matched CIFAR-10 global controls | same seed, initialization, training order and 20-epoch budget\nInherited old baseline; one adaptation seed. Projection + deterministic mean reproduces the gain.',fontsize=12);save(fig,'07_matched_global_controls')

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);gallery();cub_results();cifar_results();matched_controls()
