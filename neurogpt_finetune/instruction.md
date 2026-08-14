# neurogpt Motor6Class LOSO 训练交接

- 工作目录：`neurogpt_finetune/`
- 数据：`./AllSubjects_Epochs/{train,val,test}`（已放好，无需改动）
- 预训练权重：`./pytorch_model.bin`
- 通道矩阵：`./tMatrix_22x20_motor.npy`

运行：

```bash
cd neurogpt_finetune
bash scripts/review-finetune-motor6class-LOSO.sh
```

多卡/换卡：脚本默认只用单卡 `gpu_id=0`（`scripts/review-finetune-motor6class-LOSO.sh` 第 44 行），换卡就把这行改成对应卡号（如 `gpu_id=1`），内部通过 `CUDA_VISIBLE_DEVICES=${gpu_id}` 传给 python。

结果：
- 每折指标/预测：`fold_results_neurogpt/motor_neurogpt_fold{00..19}.json/.npz`
- 训练日志/TensorBoard：`results/motor6class_LOSO/`