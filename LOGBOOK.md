# 实验日志 LOGBOOK

> 记录每天的进展、决策与踩坑。新条目放最上面。

## 2026-06-01 — 项目脚手架搭建

- 创建 conda 环境 `flowproj` (python 3.11)，安装 CPU 版依赖。
- 建立目录结构与基础工具：
  - `src/utils/setup.py`：`set_seed` / `get_device`
  - `src/data/datasets.py`：CIFAR-10 dataloader，归一化到 [-1, 1]
  - `src/data/patchify.py`：基于 einops 的 patchify / unpatchify + 往返测试
  - `src/metrics/image_metrics.py`：psnr / ssim / lpips
  - `src/models/autoencoder.py`：小卷积自编码器（Tanh 输出，[-1, 1]）
  - `scripts/train_ae.py`：训练入口，支持 `--smoke`
- 完成首次 git 提交。
- 冒烟测试 `python scripts/train_ae.py --smoke` 通过。

### 待办
- [ ] diffusion 重建基线
- [ ] flow matching 重建基线
- [ ] 统一评测脚本与对比表
