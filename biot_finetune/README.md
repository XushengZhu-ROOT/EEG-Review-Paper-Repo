跑正式的20折（在 tmux 里）：


tmux new -s motion_st_loso
cd /path/to/biot_finetune
bash scripts/review-finetune-Motion-ST-LOSO.sh

跑完后用现成的脚本聚合结果：


python3 aggregate_loso_results.py --fold_results_dir ./fold_results_st --task motion --model st \
    --out loso_results_st.csv --cm_out loso_confusion_matrix_st.npy