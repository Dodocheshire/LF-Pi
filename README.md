# LF-CBM 的 Pi 噪声实验

在 LF-CBM 的图像、内部特征和全局特征三个位置加入可学习高斯扰动，训练噪声生成器与概念投影，再重拟合稀疏分类头。

```bash
cd /home/pengcheng/Pi-CBM/LF-Pi
python -m pip install -e .
python scripts/prepare.py --config configs/experiment/cifar10_internal.yaml
python scripts/train.py --config configs/experiment/cifar10_internal.yaml
```

- [实验报告](EXPERIMENT_REPORT.md)：训练流程和五组结果图。
- [方法与指标细节](details.md)：实现对照表、张量形状相关设置和指标定义。
- [噪声机制报告](NOISE_MECHANISM_REPORT.md)：配对训练、推理分解和背景干预实验。
- `configs/`：骨干、注入位置和实验配置。
- `pi_cbm/`：生成器、注入接口、CBM、训练与评估。
- `scripts/`：数据准备、训练、评估和绘图。
- `reports/`：指标、配置和结果图。
- `artifacts/`：缓存与检查点。

本目录是 [Pi-CBM](https://github.com/Dodocheshire/Pi-CBM) 的子模块，独立仓库为 [Dodocheshire/LF-Pi](https://github.com/Dodocheshire/LF-Pi)。
