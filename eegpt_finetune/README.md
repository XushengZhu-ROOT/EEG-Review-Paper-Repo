# EEGPT Motion 任务（20折 subject-independent LOSO）

## 数据 / 权重（都没进 git，要自己放）
- 数据：放到 `eegpt_finetune/Motiondata/{train,val,test}/*.pickle`（20通道 pickle，ella给的那个）
- 预训练权重：放到 `eegpt_finetune/eegpt_mcae_58chs_4s_large4E.ckpt`（约1GB）

## 怎么跑
```bash
cd eegpt_finetune
bash scripts/review-finetune-Motion-LOSO.sh
```
自动发现 Motiondata 里的受试者，跑 20 折（test=第i人，val=下一人，其余 train）。固定超参：lr=1e-3, wd=1e-2, bs=32, epochs=50, 全局微调（非 linear probe），不做 HPO。某折结果已存在会自动跳过，可中断重跑。

## 换机器/换卡要改的地方
- `scripts/review-finetune-Motion-LOSO.sh` 里的 `gpu_id=0`
- `linear_probe_EEGPT_Motor.py` 顶部的 `devices = [0]`（多卡/换卡号改这里）

## 结果在哪
- `fold_results_eegpt/{task}_eegpt_fold{i:02d}.npz/.json`：每折逐样本预测+指标，事后可重算，不用重跑训练
- 汇总 20 折：`python3 aggregate_loso_results.py`；单折核对：`python3 compute_metrics_from_npz.py --npz_dir fold_results_eegpt`
- `output/EEGPT_Motor_LOSO/` 是训练中间产物（含 checkpoint，20折约10GB），不进 git。