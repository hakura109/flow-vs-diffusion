# Flow Matching vs Diffusion: Image Patch Reconstruction

毕业设计项目：对比 **Diffusion** 与 **Flow Matching** 两种生成范式在图像 patch 重建任务上的表现。

## 目标

在相同的数据（CIFAR-10）、相同的骨干网络与评测协议下，公平对比 diffusion 与 flow matching
在 image patch 重建质量（PSNR / SSIM / LPIPS）上的差异。

## 工作流

- **本地（Windows, CPU）**：仅做冒烟测试（smoke test），确认代码不崩、形状和 loss 正常。
- **云端（GPU）**：真正的训练与评测。

## 环境

```bash
conda create -n flowproj python=3.11 -y
conda activate flowproj
# CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 目录结构

```
configs/        实验配置 (yaml)
src/
  data/         数据集与 patchify
  models/       网络结构（autoencoder / diffusion / flow）
  metrics/      评测指标 (psnr/ssim/lpips)
  train/        训练循环
  utils/        通用工具 (seed/device)
scripts/        可执行入口脚本
experiments/    训练输出（git 忽略）
notebooks/      探索性分析
```

## 快速验证

```bash
python scripts/train_ae.py --smoke
```

冒烟测试会取 2 张图，做一次前向 + 反向，打印输出形状与 loss。

详细进展见 [LOGBOOK.md](LOGBOOK.md)。
