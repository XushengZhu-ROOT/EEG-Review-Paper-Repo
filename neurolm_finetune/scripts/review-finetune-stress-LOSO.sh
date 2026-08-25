set -e
# =============================================================================
# Stress fine-tune launcher (17-fold subject-independent LOSO, 有验证集选模)
# =============================================================================
# 协议与 review-finetune-motor-LOSO.sh 完全一致：--fold/--n_folds 只是告诉
# train_instruction.py 用哪一折；具体的 test=subjects[fold]/val=subjects[(fold+1)%N]
# 划分逻辑在 utils.py 的 prepare_STRESS_dataset_loso() 里（新增，跟已有的
# prepare_motor_dataset_loso 同构），不需要在这个 shell 脚本里手算 subject 名字。
#
# 每个 epoch 结束后用验证集 balanced_accuracy 选最佳 epoch（train_instruction.py
# main() 里 loso_enabled 分支的既有逻辑，未改动），折末把被选中的最佳 epoch 的
# 测试集预测保存成 {task}_{model}_fold{i:02d}.npz / .json（sample_id 事后可复现
# 指标用，跟 cbramod/motor 用的是同一套 schema）。
#
# 注意：stress 17 个受试者里有 11 个只做过 increase 或只做过 normal 单一条件
# （不像 Motor 每个受试者都有全部 6 类），某折的 val/test 受试者若恰好是单一
# 类别，balanced_accuracy 仍可计算（用 sklearn balanced_accuracy_score，单类别
# 时退化为该类的 recall，不会报错），选模不受影响；折末的 AUROC/AUPRC 等指标
# 事后用 aggregate 脚本从 npz 重算时才会遇到这个情况（参考 cbramod 那边已经
# 处理过的 aggregate_loso_results_stress.py，如果 neurolm 也要出这些指标，
# 之后可以照抄同一个思路）。
#
# 超参数（lr=5e-5, wd=0, eeg_batch_size=52, text_batch_size=13）来自你给的固定
# 配置；n_gpu/gpu_id 沿用 review-finetune-stress.sh 里同一份 stress 数据本来就
# 调好的 3-GPU 设置（注释里写的"13+52是3張剩下60G的卡的極限"）。
#
# 数据目录：dataset_dir 指向 preprocessing/stress_preprocess.ipynb 新版本输出的
# neurolm_Stress_noleak_30chan_no400up_swien42（Sub<NN>_ 文件名，带 subject id），
# 不是旧的 Stress_noleak_30chan_no400up_seed_siwen42。按下面的相对路径把该文件夹
# 放到 neurolm_finetune/ 下（或改这一行指到你服务器上的实际位置）。
# text_data_dir / NeuroLM_path 跟原有 stress/motor 脚本用的是同一份资源，未改。
#
# 用法：
#   bash scripts/review-finetune-stress-LOSO.sh   (从 neurolm_finetune/ 目录下运行)
# =============================================================================
n_gpu=3
gpu_id=1,2,3

# ---- 固定參數 ----
dataset_dir=./augmented_data/neurolm_Stress_noleak_30chan_no400up_swien42
test_data_dir=/work/HHRI-AI/YW/Yirong/NeuroLM/data/text
NeuroLM_path=checkpoints/NeuroLM-B.pt

dataset=STRESS
model_name=NeuroLM                         # 影响输出文件名 {task}_{model}_fold{i}.npz/json
task_name=stress                           # 输出文件名的 task 前缀

# ---- LOSO 設定 ----
N_FOLDS=17                                  # stress 被试数（subject_edf_mapping.csv 里没有 Patient_ID=15）；跑 0..16 折

# ---- 訓練超參（固定超参数，LOSO 各折沿用同一组）----
chan_size=30
epochs=5
lr=5e-5
wd=0e+00
min_lr=5e-5
adamw_b1=0.9
adamw_b2=0.95
tbs=13                                      # text_batch_size
ibs=52                                      # eeg_batch_size

# 所有折的結果統一存到這個資料夾，方便事後彙總
results_dir=results/${dataset}/loso_lr${lr}_wd${wd}

for (( fold=0; fold<N_FOLDS; fold++ )); do
    exp_name="loso_fold$(printf '%02d' ${fold})_lr${lr}_wd${wd}"
    out_dir=checkpoints/${dataset}/${exp_name}

    if [ -f "${results_dir}/${task_name}_${model_name}_fold$(printf '%02d' ${fold}).json" ]; then
        echo "=== [fold ${fold}] already completed, skipping ==="
        continue
    fi

    echo "--- [STARTING FOLD #${fold}] ---"
    echo "Exp Name: ${exp_name}"
    echo "Out dir: ${out_dir} | Results dir: ${results_dir}"
    echo "LR: ${lr}, WD: ${wd}, Text BS: ${tbs}, EEG BS: ${ibs}"
    echo "-----------------------------------"

    CUDA_VISIBLE_DEVICES=${gpu_id} MP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=${n_gpu} train_instruction.py \
    --dataset_dir ${dataset_dir} \
    --text_data_dir ${test_data_dir} \
    --out_dir ${out_dir} \
    --results_dir ${results_dir} \
    --NeuroLM_path ${NeuroLM_path} \
    --chan_size ${chan_size} \
    --model_name ${model_name} \
    --task_name ${task_name} \
    --fold ${fold} \
    --n_folds ${N_FOLDS} \
    --eeg_batch_size ${ibs} \
    --text_batch_size ${tbs} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --min_lr ${min_lr} \
    --beta1 ${adamw_b1} \
    --beta2 ${adamw_b2} \
    --weight_decay ${wd}

    echo "--- [FOLD #${fold} FINISHED] ---"
    echo ""
done

echo "========================================"
echo "All ${N_FOLDS} LOSO folds completed."
echo "Results saved under: ${results_dir}"
echo "  ${task_name}_${model_name}_fold{00..$(printf '%02d' $((N_FOLDS-1)))}.npz / .json"
echo "========================================"
