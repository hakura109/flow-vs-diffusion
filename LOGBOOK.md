# 实验日志 LOGBOOK

> 记录每天的进展、决策与踩坑。新条目放最上面。

## 2026-06-01 — Stage 1：AE 在真实 CIFAR-10 上跑通

补完 `scripts/train_ae.py`，新增三件事：

- **完整训练**：真实 CIFAR-10（首次自动下载到 `data/`，约 170MB），MSE + Adam，
  每步 loss 写入 TensorBoard（`experiments/<时间戳>_train/tb`），训练后保存权重
  `autoencoder.pt`。
- **`--overfit` 模式**：取一个真实 batch（默认 32 张）反复训练 300 步，验证模型确实能学。
  - 结果：loss `0.3747 → 0.00156`（约 240× 下降），CPU 上约 1 分钟。✅
  - 断言：末步 loss < 初始的 10%，且 < 0.01 判为接近 0。
- **评估 + 可视化**：训练后在测试集上算 PSNR/SSIM/LPIPS 平均值，保存重建网格图
  `recon_grid.png`（上排原图、下排重建）和 `metrics.md` 指标表，全部落在
  `experiments/<时间戳>_train/`。

约定与踩坑：
- 三种模式由 flag 区分：`--smoke`（合成图）/ `--overfit`（真实单 batch）/ 默认完整训练。
- LPIPS 输入需 `[-1, 1]`，与数据归一化一致；PSNR/SSIM 内部转回 `[0, 1]` 计算。
- 完整训练只在云端 GPU 上做；本机仅用 `--overfit` 和短 epoch 验证管线不崩。

### 待办
- [ ] 云端 GPU 上完整训练 AE 并记录基线 PSNR/SSIM/LPIPS
- [ ] diffusion 重建基线
- [ ] flow matching 重建基线
- [ ] 统一评测脚本与对比表

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
