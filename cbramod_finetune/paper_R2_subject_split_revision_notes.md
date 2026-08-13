# R2 审稿意见应对与论文修改参考（Subject-Independent 分类实验）

> 基于 `Review of R2.docx`、本次 CBraMod MotorTask 受试者独立实验，以及原先随机 epoch 划分结果整理。  
> 用途：改 `IMUEEGdataset_R2` 时对照加写 Methods / Results / Discussion。

---

## 1. 审稿核心问题（与本次实验直接相关）

### Reviewer Point 1：随机 epoch 划分存在数据泄漏

- **原文问题**：`All epochs were randomly divided into training (80%) / validation (10%) / test (10%).`
- **原因**：同一连续录音的相邻 1 s epoch 高度自相关；随机划 epoch 会使近重复窗口同时出现在 train/test，准确率虚高。
- **审稿建议**：
  1. 用 **subject-independent** 划分重跑（如 leave-several-subjects-out）；
  2. **同时报告** within-subject（乐观）与 subject-independent（更诚实）两组数；
  3. 若 subject-independent 明显下降，**不算失败**，反而更有利于真实 BCI 场景。

本次我们主要完成了 Point 1 对应的实验与方法沉淀。其余 Points 2–7 见文末清单（未在本次代码实验中处理）。

---

## 2. 我们采用的方法（可写入论文 Methods）

### 2.1 任务与模型（与旧实验一致，便于对比）

| 项目 | 设定 |
|------|------|
| 任务 | 六类运动 EEG 分类 |
| 输入 | 1 s 非重叠 epoch，20 通道 |
| 骨干 | 预训练 **CBraMod**，`use_pretrained_weights=True` |
| 微调方式 | **finetune all**（`frozen=false`，非 linear probe） |
| 分类头 | `all_patch_reps` |
| 优化 | AdamW，`lr=5e-4`，`weight_decay=0.002`，`batch_size=64`，`epochs=50`，`seed=0` |
| 选模标准 | **Validation Balanced Accuracy（Val BACC）最大** |

### 2.2 旧划分（Within-subject / Random-epoch，保留作对照）

- 磁盘上 `train/val/test` 为 **按 epoch 随机** 划分（约 80/10/10）。
- **每个受试者的 epoch 同时出现在三个集合中**（已确认 20 名受试者均出现在 train/val/test）。
- 存在相邻窗口跨集合泄漏风险 → 对应审稿指出的乐观估计。

### 2.3 新划分（Subject-independent，R1）

- 先合并原 `train/val/test` 全部 epoch，再 **严格按 Subject ID** 重划。
- **单折 dry-run**（`single_fold_debug=True`），非完整 LOSO：
  - **Train**：18 人  
    `Sub01–Sub11, Sub13–Sub19`（数据中无 Sub12）
  - **Val**：`Sub20`（监控训练、按 Val BACC 选 best epoch）
  - **Test**：`Sub21`（仅最终报告；不参与选模）
- Train / Val / Test **受试者互不重叠**，从机制上消除“同人相邻 epoch 泄漏”。

### 2.4 报告指标（与审稿要求对齐）

- Balanced Accuracy（各类 recall 算术平均）
- Macro F1
- Per-class Recall
- Confusion Matrix  
- （日志中同时保留 Acc、Kappa 便于对照旧实验）

### 2.5 代码入口（方便复现，可放 Data Availability / Supplementary）

- 训练：`bash review-finetune-motortask_R1.sh`  
  - `split_mode=subject_independent`（新）  
  - 改回 `split_mode=random_epoch` 可复现旧划分
- 可视化：`bash run_visualize_motor_R1.sh` → `./viz_R1/`

---

## 3. 实验结果对比（可写入 Results）

### 3.1 主结果表（建议论文用）

选模均为 **best Val BACC** 对应的 test 指标。

| Split protocol | Best epoch (by Val BACC) | Test Acc | Test BACC | Test Kappa | Test F1* |
|----------------|--------------------------|----------|-----------|------------|----------|
| Random-epoch (within-subject, 旧) | ~47 | ~0.99 | **~0.990** | ~0.99 | ~0.99 (weighted) |
| Subject-independent 单折 (R1) | **4** | 0.500 | **0.500** | 0.400 | 0.505 (macro) |

\*R1 日志中 F1 为 **Macro F1**；旧实验为 weighted F1，对比时建议统一注明。

**一句话解读（可写 Discussion）**  
随机 epoch 划分下 CBraMod 可达约 **99%** BACC，但这反映的是 **within-subject / 近邻窗口可泄漏** 的难度；受试者独立单折下 Test BACC 降至约 **50%**，说明跨受试者泛化远难于跨 epoch 泛化。按审稿意见，应 **两组都报**，并明确 99% 不能解释为“对未见受试者”的性能。

### 3.2 R1 单折详细数字（best = ep4, Val BACC）

**Validation（Sub20）**

- Acc 0.501 / BACC **0.510** / Kappa 0.402 / Macro F1 0.485

**Test（Sub21）**

- Acc 0.500 / BACC **0.500** / Kappa 0.400 / Macro F1 0.505

### 3.3 关于“图上 20+ epoch 更高”的说明（防写错）

- 选模看的是 **蓝线 Val BACC**，最高在 **ep4 ≈ 0.510**。
- 图上约 ep21 的高峰是 **橙线 Test BACC（≈0.566）**，**不参与选模**。
- 若用 test 峰值挑模型，等于用测试集调参，审稿会认为二次泄漏。  
  论文中应写清：**model selection by validation BACC only; test evaluated once at the selected epoch.**

### 3.4 当前实验的局限（建议主动写进 Limitations）

1. **仅单折 dry-run**，尚未做完整 leave-one-subject-out / leave-several-subjects-out 交叉验证；单折结果依赖 val=`Sub20`、test=`Sub21`，方差未知。
2. 受试者编号为 Sub01–Sub11, Sub13–Sub21（无 Sub12），与审稿 Point 5 的编号问题相关，正文需统一说明。
3. 六类在未见受试者上约 chance–中等水平，适合作为 **更具挑战的公开 baseline**，而非宣称 SOTA。

---

## 4. 论文各处建议加写什么

> 以下按“该改哪一类章节 / 加什么内容”组织；你对照 R2 PDF 原有 Technical Validation / Classification 小节粘贴改写即可。

### 4.1 Methods — Data splitting / Classification protocol（必改）

**替换或并列原句**  
原文类似：“All epochs were randomly divided into 80%/10%/10% …”

**建议改成两段协议：**

1. **Within-subject (random-epoch) split**  
   - 保留原随机划分描述；  
   - **加一句**：该协议可能因相邻 epoch 自相关导致 train/test 信息泄漏，因此视为 **乐观上界**。

2. **Subject-independent split**  
   - 按受试者严格划分；本报告给出单折：18 train / 1 val / 1 test；  
   - 明确 val 用于 early-stopping / checkpoint selection，test 仅最终评估；  
   - 注明受试者列表（或附录表）；  
   - 说明后续可扩展为完整 LSSO/LOSO。

### 4.2 Methods — Model training（小改即可）

补全：预训练 CBraMod、finetune-all、超参、seed、选模指标（Val BACC）、评价指标（BACC、Macro F1、per-class recall、CM）。

### 4.3 Results — Classification benchmark（必加对照表/段）

- 增加 **Table：两种 split 的 Test BACC（及 Acc/F1/Kappa）**。  
- 可放 R1 的 test confusion matrix（`viz_R1/08_test_cm.png`）作 Figure。  
- 文中明确：~99% 来自 random-epoch；~50% 来自 subject-independent 单折。

**示例表述（英，可直接改）**

> Under a random epoch-wise split, fine-tuning CBraMod achieved ~99% test balanced accuracy. However, because adjacent 1 s windows from the same recording are highly autocorrelated, this protocol can leak near-duplicate samples across partitions. Using a subject-independent split (18/1/1 subjects for train/val/test), the same training setup yielded ~50% test balanced accuracy on the held-out subject. We therefore report both estimates: the former as an optimistic within-subject reference, and the latter as a more realistic cross-subject baseline for BCI-oriented use of this dataset.

### 4.4 Discussion / Technical Validation 解读（必加）

要点：

- 大幅下降 **验证了审稿担忧**，不是实验失败；  
- 数据集对 **跨人运动分类** 仍具挑战，适合做更严格 benchmark；  
- 99% 应解释为 **同受试者、可能泄漏条件下的可分性**，不代表跨人 BCI 性能。

### 4.5 Limitations / Future work

- 完整 subject-independent CV（多折平均 ± SD）；  
- 可选 session-/trial-level 划分作第三种协议；  
- 编号 Sub12 缺失 / Sub21 存在原因（呼应审稿 Point 5）。

### 4.6 Abstract / Highlights（若篇幅允许）

加半句：同时提供 within-subject 与 subject-independent 分类基准；跨人性能显著低于随机 epoch 划分。

---

## 5. 审稿其余条目（本次未跑实验，改稿 checklist）

| # | 问题 | 论文侧建议动作 |
|---|------|----------------|
| 2 | IMU–EEG 相关只用单 trial 却写 dataset-wide 结论 | 做分组 mean±SD，或把措辞降为 illustrative example |
| 3 | ASR `k≥5` 低于所引 10–100 范围 | 补引用/敏感性检查，或说明 noisy subset 上 ERP 仍保留 |
| 4 | Table 1 的 Slow/Fast 未指明任务 | 写明（如仅 straight walking），并写入表注 |
| 5 | 受试者编号 Sub12 缺失、Sub21 存在 | 脚注排除原因，或统一重编号；与分类实验受试者列表一致 |
| 6 | 删掉重采样句却未回答对齐 | 明确 IMU/EEG 如何对齐（插值/降采样/仅窗口级融合） |
| 7 | Fig.8 inset 轴与 caption 不符 | 改 caption 匹配实际 µV 范围 |
| 小 | Fig.9a “Straingt” 拼写 | 改为 Straight |

---

## 6. 给编辑/审稿人的回复要点（Response letter 草稿）

**对 Point 1：**

1. 我们同意随机 epoch 划分存在自相关泄漏风险。  
2. 已补充 subject-independent 协议，并 **同时报告** 原 within-subject 结果与新结果。  
3. 同一 CBraMod finetune-all 设定下：random-epoch Test BACC ≈ **99%**；subject-independent 单折 Test BACC ≈ **50%**。  
4. 选模严格基于 validation BACC；test 仅在选定 epoch 评估一次。  
5. 当前为单折 dry-run；完整 LSSO/LOSO 可作为后续工作（若本期能补一折以上更佳）。

---

## 7. 数字速查（写表时复制）

**Subject-independent（R1，finetune-all，seed=0）**

- Split: train 18 subjects / val Sub20 / test Sub21  
- Best epoch: **4**（Val BACC）  
- Test: Acc **0.500**，BACC **0.500**，Kappa **0.400**，Macro F1 **0.505**  
- 权重目录：`models_weights/MotorTask/exp_author_config_R1-all_patch_reps-all-subject_independent/`  
- 图：`viz_R1/`

**Random-epoch（旧，对照）**

- Test BACC ≈ **0.990**（best by Val BACC，约 ep47）  
- 目录：`models_weights/MotorTask/exp_author_config-all_patch_reps-all/`

---

## 8. 建议你下一步（按优先级）

1. **先改论文 Point 1**：Methods 双协议 + Results 对照表 + Discussion 解释掉点。  
2. 若时间允许：再跑 2–3 个不同 test 受试者折，报告 mean±SD（审稿更稳）。  
3. 并行处理 Points 2–7 的文字/图表澄清（多数不依赖再训练）。
