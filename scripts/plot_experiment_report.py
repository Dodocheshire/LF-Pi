"""Draw the Chinese LF-Pi pipeline and the five report result figures."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/main'
font_manager.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams.update({'font.family':'WenQuanYi Zen Hei','font.weight':500,
                    'axes.titleweight':500,'mathtext.fontset':'stix'})
COLORS = ['#8795A5','#4285C5','#46A178','#E89C45']
NAMES = ['干净 CBM','全局注入','内部注入','图像注入']


def number(value):
    if '×10⁻¹³' in value:
        return float(value.split('×')[0])*1e-13
    return float(re.search(r'[\d.]+',value).group())


def panel(ax,values,labels,title,colors=None,decimals=2):
    x=np.arange(len(values))
    bars=ax.bar(x,values,color=colors or COLORS,width=.62)
    for bar,v in zip(bars,values):
        label='≈0' if 0<v<1e-10 else f'{v:.{decimals}f}'
        ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.025,label,
                ha='center',va='bottom',fontsize=10)
    ax.set_xticks(x,labels,fontsize=10)
    ax.set_title(title,fontsize=14,pad=12)
    ax.set_ylim(0,max(values)*1.16)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)


def result_figures():
    tables=json.loads((OUT/'experiment_chart_data.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    for i,dataset in enumerate(['CIFAR-10','CIFAR-100']):
        rows=[r for r in tables[0]['rows'] if r[0]==dataset]
        observed=[rows[j] for j in [0,2,3,4]]
        for col,(metric,precision) in enumerate([('准确率（%）↑',2),('NLL ↓',4),('NEC ↓',2),('概念拟合 ↑',4)]):
            vals=[number(r[col+2]) for r in observed]
            labels=NAMES.copy();colors=COLORS.copy()
            if col==0:
                vals.append(number(rows[1][2]));labels.append('原论文');colors.append('#C6CED6')
            panel(axes[i,col],vals,labels,f'{dataset}｜{metric}',colors,precision)
            if col==0:
                axes[i,col].errorbar([4],[vals[4]],yerr=[.06 if i==0 else .12],fmt='none',ecolor='#64748B',capsize=4)
    fig.suptitle('CIFAR：CLIP RN50，三种噪声注入位置',fontsize=20)
    fig.savefig(OUT/'cifar_results.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(18,4.8),layout='constrained')
    rows=tables[1]['rows'];observed=[rows[j] for j in [0,2,3,4]]
    for col,(metric,precision) in enumerate([('准确率（%）↑',2),('NLL ↓',4),('NEC ↓',2),('概念拟合 ↑',4)]):
        vals=[number(r[col+2]) for r in observed];labels=NAMES.copy();colors=COLORS.copy()
        if col==0:
            vals.append(number(rows[1][2]));labels.append('原论文');colors.append('#C6CED6')
        panel(axes[col],vals,labels,metric,colors,precision)
        if col==0:
            axes[col].errorbar([4],[vals[4]],yerr=[.29],fmt='none',ecolor='#64748B',capsize=4)
    fig.suptitle('CUB：任务预训练 ResNet18，208 个概念',fontsize=20)
    fig.savefig(OUT/'cub_results.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5.6),layout='constrained')
    rows=tables[2]['rows'];values=[number(r[2]) for r in rows]
    labels=['论文：稠密分类器','论文：稀疏分类器','论文：LF-CBM']+NAMES
    panel(ax,values,labels,'ImageNet：Top-1 准确率（%）',
          ['#C6CED6']*3+COLORS,3)
    ax.errorbar([2],[values[2]],yerr=[.05],fmt='none',ecolor='#64748B',capsize=4)
    fig.savefig(OUT/'imagenet_accuracy.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(18,4.8),layout='constrained')
    for j,title in enumerate(['NLL ↓','NEC ↓','概念漂移 MSE ↓','概念拟合 ↑']):
        panel(axes[j],[number(r[j+1]) for r in tables[3]['rows']],NAMES,title,decimals=4 if j!=1 else 2)
    fig.suptitle('ImageNet：50,000 张官方验证图',fontsize=20)
    fig.savefig(OUT/'imagenet_metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(21,4.8),layout='constrained')
    for j,title in enumerate(['准确率（%）↑','NLL ↓','NEC ↓','概念漂移 MSE ↓','概念拟合 ↑']):
        panel(axes[j],[number(r[j+1]) for r in tables[4]['rows']],NAMES,title,decimals=2 if j in [0,2] else 4)
    fig.suptitle('Places365：全量训练，36,500 张官方验证图',fontsize=20)
    fig.savefig(OUT/'places365_results.png',dpi=160);plt.close(fig)


def pipeline():
    from torchvision.datasets import CIFAR10
    image=np.asarray(CIFAR10('/home/pengcheng/VLG-CBM/datasets',train=False,download=False)[0][0])
    fig,ax=plt.subplots(figsize=(22,15))
    ax.set(xlim=(0,22),ylim=(0,15));ax.axis('off')
    blue,pink,green='#DFEDF9','#F8DFE7','#E3F1DE'
    def txt(x,y,s,size=13,**kw):ax.text(x,y,s,ha='center',va='center',fontsize=size,**kw)
    def box(x,y,w,h,s,color,size=13):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#334155',lw=1.3,zorder=3));txt(x+w/2,y+h/2,s,size,zorder=4)
    def arrow(a,b,color='#334155',dashed=False,bend=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color=color,lw=1.5,
            linestyle='--' if dashed else '-',connectionstyle=f'arc3,rad={bend}',zorder=2))
    def picture(x,y,w,label):
        ax.imshow(image,extent=(x,x+w,y,y+w),interpolation='nearest',aspect='auto',zorder=3)
        ax.add_patch(Rectangle((x,y),w,w,fill=False,edgecolor='#334155',zorder=4));txt(x+w/2,y-.25,label,12)
    def vector(x,y,w,h,label):
        for j,v in enumerate([.7,.15,.85,.45,.2,.95,.5,.3]):
            ax.add_patch(Rectangle((x+j*w/8,y),w/10,h*v,facecolor='#4D91BA',zorder=3))
        txt(x+w/2,y-.25,label,12)
    for y,h,title in [(10.7,3.7,'（a）干净 CBM：按顺序拟合'),(6.3,4.05,'（b）三种注入位置：每次实验选择一处'),(.3,5.65,'（c）噪声适配与分类头重拟合')]:
        ax.add_patch(Rectangle((.25,y),21.5,h,fill=False,edgecolor='#334155',lw=1.4))
        ax.text(.55,y+h-.25,title,fontsize=20,va='top')
    # Independent teacher and backbone streams meet at projection training.
    box(.8,12.0,3.3,1.05,'CLIP 图像／文本编码器\n生成概念相似度目标',blue)
    box(4.7,12.0,2.55,1.05,'① 候选概念筛选\n依据 CLIP 目标',green)
    box(7.9,12.0,2.7,1.05,'② 训练概念投影\n图像特征 → 概念值',pink)
    box(11.25,12.0,2.9,1.05,'③ 再筛选概念\n标准化概念值',green)
    box(14.8,12.0,2.7,1.05,'④ 固定投影\nglm_saga 拟合稀疏头',green)
    box(18.15,12.0,2.75,1.05,'干净 CBM\n投影 ＋ 稀疏分类头',green)
    for a,b in [(4.1,4.7),(7.25,7.9),(10.6,11.25),(14.15,14.8),(17.5,18.15)]:arrow((a,12.52),(b,12.52))
    box(6.8,11.05,3.0,.62,'冻结骨干的图像特征',blue,12)
    arrow((8.3,11.68),(8.3,11.98))
    txt(15.7,11.3,'投影先拟合；分类头随后独立拟合',13)
    # RN50 visual path, spatial tensor drawn as a stack, global tensor as a vector.
    picture(.85,7.05,1.65,'预处理图像')
    box(4.0,7.45,2.45,1.05,'CLIP RN50\n卷积主干（冻结）',blue)
    for j in range(3):
        grid=np.arange(49).reshape(7,7)/(49+j*10)
        ax.imshow(grid,extent=(7.3+j*.12,8.7+j*.12,7.1+j*.12,8.5+j*.12),cmap='Blues',aspect='auto',zorder=3+j)
    txt(8.13,6.79,'空间特征：2048×7×7',12)
    box(10.55,7.45,2.25,1.05,'注意力池化\n冻结',blue)
    vector(13.85,7.15,1.55,1.1,'全局向量：1024 维')
    box(17.0,7.45,1.7,1.05,'概念投影',pink)
    box(19.35,7.45,1.7,1.05,'稀疏分类头',green)
    for a,b in [(2.5,4),(6.45,7.25),(9,10.55),(12.8,13.8),(15.4,17),(18.7,19.35)]:arrow((a,7.96),(b,7.96))
    for x,name,c in [(3.2,'图像注入','#E89C45'),(9.7,'内部特征注入','#46A178'),(16.15,'全局特征注入','#4285C5')]:
        ax.add_patch(Circle((x,7.96),.23,facecolor='white',edgecolor=c,lw=2,zorder=5));txt(x,7.96,'＋',18,color=c,zorder=6)
        txt(x,9.1,name,14,color=c)
        arrow((x,8.88),(x,8.22),c)
    # Concept conditioning is drawn as a text card rather than an image feature.
    box(.85,3.35,3.0,1.55,'固定概念文本向量\n“翅膀”“轮子”“条纹”…\nCLIP ViT-B/16：K×512','#FFF3D7',12)
    box(4.55,3.6,3.0,1.15,'噪声生成器（更新）\n视觉 Query＋文本 Key / Value',pink,12)
    arrow((3.85,4.13),(4.55,4.13))
    box(4.55,4.94,3.0,.38,'所选位置的视觉 token',blue,12)
    arrow((6.05,4.93),(6.05,4.77))
    box(8.15,3.6,2.6,1.15,'预测均值 μ 与尺度 σ\n采样 μ＋σ⊙ξ',pink,13)
    arrow((7.55,4.13),(8.15,4.13))
    arrow((9.45,4.76),(9.7,7.7),'#BF3556',True)
    txt(11.12,5.37,'送到本次选择的注入位置',12,color='#BF3556')
    box(12.0,3.6,2.75,1.15,'概念投影（更新）\n得到扰动后的概念值',pink)
    box(15.45,3.6,2.65,1.15,'原分类头（固定）\n计算分类损失',green)
    box(18.7,3.6,2.3,1.15,'分类＋概念约束\n＋噪声能量约束','#FFF3D7',12)
    arrow((14.75,4.13),(15.45,4.13));arrow((18.1,4.13),(18.7,4.13))
    arrow((17.85,7.4),(13.4,4.78),bend=.08)
    # The loss feeds both trainable modules; no gradient update to the sparse head.
    ax.plot([19.8,19.8,6.05],[3.56,3.1,3.1],color='#BF3556',linestyle='--',lw=1.5)
    arrow((6.05,3.1),(6.05,3.57),'#BF3556',True)
    arrow((13.4,3.1),(13.4,3.57),'#BF3556',True)
    txt(11.0,2.69,'适配阶段：同时更新生成器和投影；骨干与分类头固定',14,color='#BF3556')
    box(3.2,1.0,3.2,.85,'适配完成后固定\n生成器 ＋ 概念投影',pink,12)
    box(7.1,1.0,3.45,.85,'每张训练图采样一次\n缓存固定概念值',blue,12)
    box(11.25,1.0,3.6,.85,'glm_saga 单独重拟合\n稀疏分类头',green,12)
    box(15.55,1.0,3.1,.85,'最终加噪 CBM\n评估分类与概念指标',green,12)
    for a,b in [(6.4,7.1),(10.55,11.25),(14.85,15.55)]:arrow((a,1.43),(b,1.43))
    fig.suptitle('LF-Pi：概念条件噪声与顺序拟合训练',fontsize=27,y=.985)
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.95)
    fig.savefig(OUT/'pipeline.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    result_figures()
    pipeline()
