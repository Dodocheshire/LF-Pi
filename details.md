# LF-Pi 方法与指标细节

## 方法实现对照

| 环节 | 论文中组件 | 当前实现 | 代码 |
| --- | --- | --- | --- |
| 概念与文本目标 | LF-CBM先建立、筛选概念集合，再用CLIP文本编码器提供概念表示 | CIFAR-10/100使用固定概念列表和冻结的CLIP ViT-B/16文本表示；CUB导入已训练的LF-CBM概念列表及参数 | `configs/concepts/`、`pi_cbm/data.py`、`pi_cbm/models/import_lfcbm.py` |
| CBM基线 | LF-CBM训练图像特征到概念的投影，并使用稀疏线性分类头 | CIFAR-10/100从头训练CLIP RN50特征上的概念投影，按官方激活与可解释性阈值筛选概念，再使用官方`glm_saga`拟合分类头 | `pi_cbm/training/lfcbm.py`、`pi_cbm/models/cbm.py`、`pi_cbm/vendor/lfcbm/` |
| 噪声生成 | PiNI以视觉特征为条件预测对角高斯分布的均值和尺度，再进行重参数采样 | 实现`NoiseGenerator`，默认同时预测均值与标准差；概念条件交叉注意力读取当前图像特征和固定的概念文本向量。概念文本向量本身冻结，注意力中的Query、Key、Value投影可训练| `pi_cbm/generators/noise.py` |
| 注入位置 | PiNI可在原始图像或视觉编码路径的不同位置加入扰动 | 一套注入接口支持`target_image`、`target_internal`、`target_global`：分别作用于预处理图像、骨干中间特征、骨干输出的全局特征。随后沿原骨干/CBM路径计算概念和分类结果。 | `pi_cbm/injections/sites.py`、`pi_cbm/models/backbones.py` |
| 适配训练 | PiNI通过下游监督信号训练噪声；LF-CBM用投影和稀疏头维持概念分类路径 | 统一采用`finetune_projection`：训练生成器和概念投影，损失包含分类交叉熵、与干净概念的均方误差约束、噪声能量约束，以及与CLIP概念目标的`cos_cubed`对齐；按适配验证集NLL选模，再用官方`glm_saga`重拟合稀疏分类头。CIFAR运行还采用梯度范数裁剪(之前有梯度不稳定现象) | `pi_cbm/training/adapt.py`、`pi_cbm/training/lfcbm.py` |
| 评估与复现 | [VLG-CBM](https://arxiv.org/abs/2408.01432)第4.2节提出NEC衡量分类头每类使用的有效概念数 | 统计准确率、NLL、标准化概念漂移、与教师概念的相似度及NEC | `pi_cbm/evaluation/inference.py`、`pi_cbm/pipeline.py`、`reports/`中的JSON/YAML |

## 指标定义

NEC为：对最终分类头权重$W\in\mathbb R^{C\times K}$，以$|W_{ij}|>10^{-5}$视为一个有效“类别—概念”连接，按类别取平均

$$
\mathrm{NEC}(W)=\frac{1}{C}\sum_{i=1}^{C}\sum_{j=1}^{K}\mathbf{1}\{|W_{ij}|>10^{-5}\}
$$

**NLL（负对数似然）**：所有测试图损失的平均值

**CLIP概念拟合`cos_cubed`**：如果没有逐图的真实概念标注，可以用冻结的CLIP图像编码器与概念文本编码器计算相似度，作为概念层的代理监督信号。对每个概念，比较它在全部测试图上的CLIP相似度与CBM概念输出的变化趋势：两列数值分别减去均值、取三次方，再计算余弦相似度，越高表示越贴近CLIP监督信号。

## 概念筛选与训练设置

CIFAR 基线投影选择、概念可解释性筛选和分类头验证使用配置 `validation_split: test`；基线使用完整 50,000 张训练图，适配训练与分类头重拟合使用其中 45,000 张，另 5,000 张用于适配验证。

ImageNet 使用 4,751 项初始概念和官方筛选门槛。每类 200 张训练图的首次筛选保留 4,440 项，投影训练后的第二次筛选保留 4,141 项；原论文使用完整训练集，保留 4,505 项。

概念条件使用冻结的 CLIP ViT-B/16 文本向量；生成器分别学习视觉 Query 与文本 Key、Value 的投影，随后计算交叉注意力。
