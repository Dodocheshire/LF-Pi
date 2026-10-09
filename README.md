# LF-CBM 的 Pi 噪声实验

原仓库实验完整迁入此目录，包名仍为 `pi_cbm`。运行命令前先进入本目录；配置里的相对路径以本目录为工作目录。

```bash
cd /home/pengcheng/Pi-CBM/LF-Pi
python -m pip install -e '.[dev]'
python scripts/prepare.py --config configs/experiment/cifar10_internal.yaml
python scripts/train.py --config configs/experiment/cifar10_internal.yaml
python -m pytest -q
```

- [实验报告](EXPERIMENT_REPORT.md)：CIFAR-10、CIFAR-100、CUB 的准确率、NLL、NEC 和概念对齐。
- `configs/`：骨干、注入位置、服务器数据集与实验配置。
- `pi_cbm/`：噪声生成器、注入接口、CBM、训练与评估。
- `scripts/`：准备数据、训练、评估及服务器批量运行。
- `reports/`：归档指标和对应配置。
- `artifacts/`：原缓存与检查点，整体移动，未重新生成。
- [迁移记录](migration.json)：迁移前源文件哈希、Git 状态与更新的路径列表。

服务器配置和 ImageNet 下载脚本中，原指向 `/home/pengcheng/Pi-CBM/artifacts` 和 `/home/pengcheng/Pi-CBM/scripts` 的运行路径已调整到 `LF-Pi/`。归档报告和检查点内部的历史路径保留原值。

此目录作为 [Pi-CBM](https://github.com/Dodocheshire/Pi-CBM) 的 submodule 管理，独立远端为 [Dodocheshire/LF-Pi](https://github.com/Dodocheshire/LF-Pi)。保留迁移前主仓库的提交历史，包括 ImageNet 与 Places365 全量实验更新。特征和训练权重保存在忽略的 `artifacts/` 中。
