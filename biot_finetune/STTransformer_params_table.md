# 2. STTransformer 参数量（程序真实输出）

**配置来源**: `STTransformer-review-finetune-SEED.sh` → `run_multiclass_supervised.py`  
**真实配置**: SEED-ST, in_channels=62, n_classes=6, sampling_rate=250, sample_length=4s  
**说明**: depth=4（4 层 TransformerEncoder），与同事截图中 3 层不同；数值由 `count_sttransformer_params.py` 程序遍历模型计算得出。

---

## 参数表格（Component | Params）

| Component | Params |
| :-------- | :----- |
| ChannelAttention block (LayerNorm + CA + Dropout, ResidualAdd) | 14,090 |
| PatchSTEmbedding (2×Conv1d + BatchNorm + Rearrange) | 305,728 |
| TransformerEncoder (4 layers) | 3,159,040 |
| **Backbone subtotal** | **3,478,858** |
| ClassificationHead (ELU + Linear 256→6) | 1,542 |
| **Total** | **3,480,400** |

**Buffers**: 129 (BatchNorm running stats)  
**Grand total**: 3,480,529

---

## TransformerEncoder 各层参数量（每层相同，程序 output）

| Layer | Params |
| :---- | :----- |
| Layer 0 | 789,760 |
| Layer 1 | 789,760 |
| Layer 2 | 789,760 |
| Layer 3 | 789,760 |

---

## 与同事截图的差异

- **depth**: 实际运行为 4 层，同事表格为 3 层  
- **ChannelAttention**: 14,090（n_channels=62，非 16）  
- **PatchSTEmbedding**: 305,728（首层 Conv1d 为 62→64，因 SEED 为 62 通道）  
- **ClassificationHead**: 1,542（ELU + Linear 256→6，与同事一致）
