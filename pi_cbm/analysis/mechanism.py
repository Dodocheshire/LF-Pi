"""Exact distribution counterfactuals, without changing archived training code."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import torch
from torch.nn import functional as F
from pi_cbm.injections.sites import to_tokens,from_tokens

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


@contextmanager
def gpu_lease(label):
    path=ROOT.parent/'D-Pi/artifacts/resources/gpu_work.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def distribution(model,inputs):
    if model.generator is None:return torch.zeros_like(inputs),torch.zeros_like(inputs)
    if model.injection.cache_key=='images':
        pooled=F.adaptive_avg_pool2d(inputs,(model.generator.image_grid,)*2)
        mean,scale=model.generator.distribution(to_tokens(pooled))
        mean=F.interpolate(from_tokens(mean,pooled.shape),size=inputs.shape[-2:],mode='bilinear')
        scale=F.interpolate(from_tokens(scale,pooled.shape),size=inputs.shape[-2:],mode='bilinear')
        return mean,scale
    mean,scale=model.generator.distribution(to_tokens(inputs))
    return from_tokens(mean,inputs.shape),from_tokens(scale,inputs.shape)


def counterfactual(model,inputs,mode='mean',samples=8,generator=None):
    mean,scale=distribution(model,inputs)
    if mode=='plain':draws=inputs.unsqueeze(0)
    elif mode=='mean':draws=(inputs+mean).unsqueeze(0)
    elif mode in ['sample','scale_only']:
        eps=torch.randn((samples,*inputs.shape),device=inputs.device,dtype=scale.dtype,generator=generator)
        draws=inputs.unsqueeze(0)+(mean.unsqueeze(0) if mode=='sample' else 0)+scale.unsqueeze(0)*eps
    else:raise ValueError(mode)
    shape=draws.shape;features=model.suffix(draws.flatten(0,1))
    concepts=model.cbm.concepts(features).reshape(shape[0],shape[1],-1)
    logits=model.cbm.head(concepts);probabilities=logits.softmax(-1)
    return probabilities.mean(0),concepts.mean(0),{'mean':mean,'scale':scale,'draws':draws,'sample_probabilities':probabilities}


def classification(probabilities,labels):
    confidence,pred=probabilities.max(1);correct=pred.eq(labels);ece=0.
    for lo,hi in zip(torch.linspace(0,1,16)[:-1],torch.linspace(0,1,16)[1:]):
        selected=(confidence>lo)&(confidence<=hi)
        if selected.any():ece+=float(selected.float().mean())*abs(float(correct[selected].float().mean())-float(confidence[selected].mean()))
    return {'accuracy':float(correct.float().mean()),'correct':int(correct.sum()),'n':len(labels),
            'nll':float(-probabilities[torch.arange(len(labels)),labels].clamp_min(1e-12).log().mean()),'ece15':ece}


def nec(cbm):return float((cbm.head.weight.abs()>1e-5).sum())/cbm.head.out_features


def normalized_energy(mean,scale,feature_scale):
    shape=(1,-1)+(1,)*(mean.ndim-2);unit=feature_scale.reshape(shape)
    dims=tuple(range(1,mean.ndim));m=(mean/unit).square().mean(dims);v=(scale/unit).square().mean(dims)
    return m,v
