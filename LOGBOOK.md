# 实验日志 LOGBOOK

> 记录每天的进展、决策与踩坑。新条目放最上面。

## 2026-06-01 — FFHQ-64 数据管道

- `src/data/datasets.py` 新增 `FlatImageDataset` + `get_ffhq64_dataset/dataloader`：
  从本地图片文件夹（ImageFolder 风格，无需类别子目录）递归读图，短边 Resize→
  CenterCrop 到 64×64，归一化到 [-1,1]。接口与 `get_cifar10_dataloader` 一致
  （FFHQ 无内置 train/test 划分，`train` 仅决定 shuffle/drop_last）。
- `scripts/make_ffhq64_placeholders.py`：在拿到真实数据前生成几十张 64×64 占位图，
  让管道可跑通；真实 FFHQ-64 放进 `data/ffhq64/` 即可替换，loader 不用改。
- 验证：48 张占位图 → batch `(16,3,64,64)`，范围 [-1.000, 0.992]。✅
- 注：`data/` 已 gitignore，图片不入库。

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
