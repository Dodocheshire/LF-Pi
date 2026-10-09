"""PiNI-style real LF-Pi images plus paired foreground/background diagnostics."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize,CenterCrop,ToTensor,InterpolationMode
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pi_cbm.analysis.mechanism import ROOT,sha,write_json,gpu_lease,distribution,counterfactual,classification,nec,normalized_energy
from pi_cbm.data import stratified_indices,digest
from pi_cbm.evaluation.checkpoint import restore_model
from pi_cbm.models.cbm import ConceptBottleneck
from pi_cbm.models.backbones import load_backbone,target_preprocess

RUNS={'target_image':'artifacts/cub-image-runs/20260926-075413-183411-cub_image_finetune_projection',
      'target_internal':'artifacts/cub-position-runs/20260926-081008-182344-cub_internal_finetune_projection',
      'target_global':'artifacts/cub-position-runs/20260926-080816-170309-cub_global_finetune_projection'}
ANNO=ROOT.parent/'D-Pi/artifacts/annotations'
OUT=ROOT/'artifacts/mechanism_v1/cub';REPORT=ROOT/'reports/noise_mechanism_v1'


def historical_sources():
    root=Path('/home/pengcheng/VLG-CBM/datasets/CUB')
    sources={s:ImageFolder(root/s) for s in ['train','test']};removed={}
    for name,source in sources.items():
        removed[name]=[];keep=[]
        for path,label in source.samples:
            with Image.open(path) as image:
                if image.mode=='RGB':keep.append((path,label))
                else:removed[name].append(path)
        source.samples=keep;source.imgs=keep;source.targets=[label for _,label in keep]
    rng=np.random.default_rng(0);val,available=stratified_indices(sources['train'].targets,5,rng)
    select,_=stratified_indices(np.asarray(sources['train'].targets)[available],0,rng)
    test,_=stratified_indices(sources['test'].targets,0,rng)
    indices={'train':available[select],'val':val,'test':test};split_hash=digest({k:v.tolist() for k,v in indices.items()})
    for run in RUNS.values():
        expected=json.loads((ROOT/run/'metrics.json').read_text())['manifest']['split_sha256']
        if expected!=split_hash:raise RuntimeError('Historical RGB-only CUB split reconstruction failed')
    return sources,indices,removed,split_hash


class CubImages(Dataset):
    def __init__(self,source,indices,masks=True):
        self.source=source;self.indices=list(map(int,indices));self.transform=target_preprocess('resnet18_cub');self.masks=masks
        self.mask_transform=lambda mask:ToTensor()(CenterCrop(224)(Resize(256,interpolation=InterpolationMode.NEAREST)(mask)))>0.5
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        sid=self.indices[index];path,label=self.source.samples[sid]
        with Image.open(path) as image:target=self.transform(image.convert('RGB'))
        mask=torch.zeros(1,224,224,dtype=torch.bool)
        if self.masks:
            mask_path=ANNO/'segmentations'/Path(path).parent.name/(Path(path).stem+'.png')
            with Image.open(mask_path) as original_mask:mask=self.mask_transform(original_mask.convert('L'))
        return target,label,mask,sid


def original_cbm():
    path=ROOT/'artifacts/pretrained/lfcbm-cub';names=(path/'concepts.txt').read_text().splitlines()
    p={k:torch.load(path/f'{k}.pt',weights_only=True,map_location='cpu') for k in ['W_c','W_g','b_g','proj_mean','proj_std']}
    cbm=ConceptBottleneck(p['W_c'].shape[1],len(names),len(p['b_g']))
    cbm.load_state_dict({'projection.weight':p['W_c'],'head.weight':p['W_g'],'head.bias':p['b_g'],
                        'concept_mean':p['proj_mean'].flatten(),'concept_std':p['proj_std'].flatten()})
    return cbm.cuda().eval().requires_grad_(False),names


def true_class_cam(backbone,cbm,images,labels):
    feature_map=backbone.suffix[0](backbone.prefix(images))
    effective=(cbm.head.weight[labels]/cbm.concept_std)@cbm.projection.weight
    cam=(feature_map*effective[:,:,None,None]).sum(1).relu()
    return F.interpolate(cam[:,None],size=(224,224),mode='bilinear',align_corners=False)[:,0]


def foreground_mass(cam,mask):
    return (cam*mask[:,0]).sum((1,2))/cam.sum((1,2)).clamp_min(1e-20)


def background_intervention(images,masks,kind):
    if kind=='neutral_bg':return torch.where(masks,images,torch.zeros_like(images))
    if kind=='neutral_fg':return torch.where(masks,torch.zeros_like(images),images)
    if kind!='swap_bg':raise ValueError(kind)
    # Fixed cyclic donors; erase donor foreground with its own background median.
    # This preserves receiver foreground exactly; synthetic seams remain a limitation.
    donors=images.roll(1,0).clone();donor_masks=masks.roll(1,0)
    for j in range(len(donors)):
        bg=~donor_masks[j,0]
        fill=donors[j,:,bg].median(1).values if bg.any() else torch.zeros(3,device=images.device)
        donors[j]=torch.where(donor_masks[j],fill[:,None,None],donors[j])
    result=torch.where(masks,images,donors)
    if not torch.equal(result[masks.expand_as(images)],images[masks.expand_as(images)]):raise RuntimeError('Background intervention changed foreground')
    return result


def js_divergence(p,q):
    mid=(p+q)/2
    return .5*((p*(p.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(1)+(q*(q.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(1))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=8);parser.add_argument('--diagnostic-n',type=int,default=800);args=parser.parse_args()
    torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources,indices,removed,split_hash=historical_sources()
    rng=np.random.default_rng(20261009);diagnostic_ids=rng.choice(len(indices['test']),size=min(args.diagnostic_n,len(indices['test'])),replace=False)
    diagnostic_ids.sort();selected_indices=indices['test'][diagnostic_ids]
    # Gallery chosen by fixed class IDs and visible mask area only, never prediction success.
    gallery_rows=[];validation=CubImages(sources['train'],indices['val'])
    for label in [0,24,49,74,99,124]:
        for j,sid in enumerate(validation.indices):
            if sources['train'].targets[sid]!=label:continue
            image,y,mask,_=validation[j];area=float(mask.float().mean())
            if .05<=area<=.8:gallery_rows.append((image,y,mask,sid));break
    gallery={name:torch.stack([row[i] for row in gallery_rows]) if i in [0,2] else torch.tensor([row[i] for row in gallery_rows]) for i,name in enumerate(['images','labels','masks','indices'])}
    protocol={'source_sha256':sha(__file__),'helper_sha256':sha(ROOT/'pi_cbm/analysis/mechanism.py'),
              'checkpoint_sha256':{site:sha(ROOT/run/'checkpoint.pt') for site,run in RUNS.items()},
              'historical_split_hash':split_hash,'historical_rgb_only_counts':{k:len(v) for k,v in indices.items()},
              'excluded_non_rgb':removed,'diagnostic_positions':diagnostic_ids.tolist(),'diagnostic_n':len(diagnostic_ids),
              'samples':args.samples,'gallery_source':'fixed validation classes and mask-area criterion, no success selection',
              'scope':'Read-only old-checkpoint diagnostics, not independent new performance acceptance; train-only attribute thresholds; masks used only for evaluation.',
              'test_used_for_new_training':False,'no_new_training':True}
    guard=OUT/'protocol.json'
    if guard.exists() and json.loads(guard.read_text())!=protocol:raise RuntimeError('Diagnostic protocol changed; use new version')
    write_json(protocol,guard);torch.save({'sources':{s:source.samples for s,source in sources.items()},'indices':{k:torch.tensor(v) for k,v in indices.items()}},OUT/'dataset.pt')
    with gpu_lease('LF-Pi CUB noise mechanism'):
        backbone=load_backbone({'name':'resnet18_cub'},'cuda');base,names=original_cbm()
        # Train-only baseline concept values for fixed native-attribute thresholds.
        train_features=[];train_scores=[];train_ids=[]
        for images,labels,mask,sid in DataLoader(CubImages(sources['train'],indices['train'],False),batch_size=128,num_workers=4):
            features=backbone(images.cuda());train_features.append(features.cpu());train_scores.append(base.concepts(features).cpu());train_ids.append(sid)
        torch.save({'features':torch.cat(train_features),'concepts':torch.cat(train_scores),'indices':torch.cat(train_ids),'names':names},OUT/'training_reference.pt')
        rows=[];all_predictions={};base_clean=[];base_concepts=[]
        loader=DataLoader(CubImages(sources['test'],selected_indices),batch_size=32,shuffle=False,num_workers=4)
        for images,labels,masks,sids in loader:
            z=base.concepts(backbone(images.cuda()));base_clean.append(base.head(z).softmax(-1).cpu());base_concepts.append(z.cpu())
        baseline_probability=torch.cat(base_clean);baseline_concepts=torch.cat(base_concepts)
        diagnostic_labels=torch.tensor([sources['test'].targets[int(i)] for i in selected_indices])
        rows.append({'site':'baseline','mode':'clean','condition':'original',**classification(baseline_probability,diagnostic_labels),'nec':nec(base)})
        all_predictions['baseline/clean/original']={'probabilities':baseline_probability,'concepts':baseline_concepts}
        baseline_changed={kind:[] for kind in ['neutral_bg','neutral_fg','swap_bg']};baseline_focus=[]
        for images,labels,masks,sids in loader:
            images=images.cuda();masks=masks.cuda();labels=labels.cuda()
            baseline_focus.append(foreground_mass(true_class_cam(backbone,base,images,labels),masks).cpu())
            for kind in baseline_changed:
                edited=background_intervention(images,masks,kind);z=base.concepts(backbone(edited));baseline_changed[kind].append((base.head(z).softmax(-1).cpu(),z.cpu()))
        for kind,batches in baseline_changed.items():
            probs=torch.cat([a for a,b in batches]);z=torch.cat([b for a,b in batches]);all_predictions[f'baseline/clean/{kind}']={'probabilities':probs,'concepts':z}
            rows.append({'site':'baseline','mode':'clean','condition':kind,**classification(probs,diagnostic_labels),'nec':nec(base),
                         'paired_js_mean':float(js_divergence(baseline_probability,probs).mean()),'concept_delta_rms':float((z-baseline_concepts).square().mean().sqrt())})
        gallery['baseline_cam']=true_class_cam(backbone,base,gallery['images'].cuda(),gallery['labels'].cuda()).cpu()
        gallery['baseline_probabilities']=base.head(base.concepts(backbone(gallery['images'].cuda()))).softmax(-1).cpu()
        for site,run in RUNS.items():
            saved=torch.load(ROOT/run/'checkpoint.pt',weights_only=True,map_location='cpu')
            if saved['concepts']!=names:raise RuntimeError('Concept bank changed between baseline and adaptation')
            suffix=backbone if site=='target_image' else (backbone.suffix if site=='target_internal' else torch.nn.Identity())
            model=restore_model(saved,'cuda',suffix);predictions={};energies=[];spatial=[]
            for kind in ['original','neutral_bg','neutral_fg','swap_bg']:
                for mode in ['plain','mean','sample']:
                    batches=[];concept_batches=[];focus=[];generator=torch.Generator(device='cuda').manual_seed(20261009)
                    for images,labels,masks,sids in loader:
                        images=images.cuda();masks=masks.cuda();labels=labels.cuda()
                        if kind!='original':images=background_intervention(images,masks,kind)
                        inputs=images if site=='target_image' else (backbone.prefix(images) if site=='target_internal' else backbone(images))
                        probabilities,z,extra=counterfactual(model,inputs,mode,args.samples,generator)
                        batches.append(probabilities.cpu());concept_batches.append(z.cpu())
                        if kind=='original' and mode=='mean':
                            me,ve=normalized_energy(extra['mean'],extra['scale'],model.generator.feature_scale);energies.extend(zip(me.cpu().tolist(),ve.cpu().tolist()))
                            if site=='target_image':
                                std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
                                mu_rms=(extra['mean']*std).square().mean(1).sqrt();sigma_rms=(extra['scale']*std).square().mean(1).sqrt();mask=masks[:,0]
                                for j in range(len(images)):
                                    spatial.append({'mean_fg':float(mu_rms[j][mask[j]].mean()),'mean_bg':float(mu_rms[j][~mask[j]].mean()),
                                                    'sigma_fg':float(sigma_rms[j][mask[j]].mean()),'sigma_bg':float(sigma_rms[j][~mask[j]].mean())})
                                focus.append(foreground_mass(true_class_cam(backbone,model.cbm,images+extra['mean'],labels),masks).cpu())
                    p=torch.cat(batches);c=torch.cat(concept_batches);key=f'{site}/{mode}/{kind}';all_predictions[key]={'probabilities':p,'concepts':c};predictions[(mode,kind)]=(p,c)
                    row={'site':site,'mode':mode,'condition':kind,**classification(p,diagnostic_labels),'nec':nec(model.cbm)}
                    if kind!='original':
                        orig_p,orig_c=predictions[(mode,'original')];row.update(paired_js_mean=float(js_divergence(orig_p,p).mean()),concept_delta_rms=float((c-orig_c).square().mean().sqrt()))
                    if focus:row['true_class_cam_foreground_mass']=float(torch.cat(focus).mean())
                    rows.append(row);print(site,mode,kind,round(100*row['accuracy'],3),flush=True)
                    write_json({'phase':'evaluate','site':site,'mode':mode,'condition':kind},OUT/'status.json')
            if site=='target_image':
                gx=gallery['images'].cuda();g=torch.Generator(device='cuda').manual_seed(20261009)
                mean,scale=distribution(model,gx);sample=gx+mean+scale*torch.randn(gx.shape,generator=g,device='cuda')
                gallery.update(mean=mean.cpu(),scale=scale.cpu(),sample=sample.cpu(),mean_probabilities=model.cbm.head(model.cbm.concepts(backbone(gx+mean))).softmax(-1).cpu(),
                               sample_probabilities=model.cbm.head(model.cbm.concepts(backbone(sample))).softmax(-1).cpu(),
                               mean_cam=true_class_cam(backbone,model.cbm,gx+mean,gallery['labels'].cuda()).cpu())
                torch.save(gallery,OUT/'gallery.pt')
            write_json({'site':site,'mean_energy':float(np.mean([m for m,v in energies])),'variance_energy':float(np.mean([v for m,v in energies])),
                        'spatial':spatial,'checkpoint_sha256':sha(ROOT/run/'checkpoint.pt')},OUT/f'{site}_distribution.json')
        torch.save({'predictions':all_predictions,'labels':diagnostic_labels,'indices':torch.tensor(selected_indices),'baseline_cam_focus':torch.cat(baseline_focus)},OUT/'predictions.pt')
        write_json({'rows':rows,'protocol':protocol},OUT/'summary.json');write_json({'phase':'complete','rows':len(rows)},OUT/'status.json')
        print('CUB mechanism diagnostics complete',len(rows),flush=True)


if __name__=='__main__':main()
