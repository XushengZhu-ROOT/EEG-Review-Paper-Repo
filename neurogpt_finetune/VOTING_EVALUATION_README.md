# 投票评估功能说明

## 功能概述

实现了基于视频级别的投票评估机制，用于SEED emotion 7分类任务。评估流程：

1. **视频级别的多数投票**：对同一视频的所有4秒chunks的预测结果进行投票
2. **平票处理**：
   - 如果真实标签在平票候选中 → 得分0.5（半对）
   - 如果真实标签不在平票候选中 → 得分0.0（错误）
3. **Subject级别准确率**：计算每个subject的80个视频的准确率

## 实现细节

### 1. Emotion7ClassDataset修改

- 在`get_trials_all()`中保存`epoch_id`信息
- `epoch_id`格式：`subject_X_video_index_Y_chunkZZZ`
- 从`epoch_id`中提取`subject_id`和`video_index`

### 2. 投票评估函数 (`compute_voting_metrics`)

位置：`src/trainer/make.py`

功能：
- 按`(subject_id, video_index)`分组
- 对每个视频的所有chunks进行多数投票
- 计算视频级别准确率和BACC
- 计算每个subject的准确率

### 3. 测试集投票评估

在`train_gpt.py`中，测试集预测后自动进行投票评估：
- 保存投票评估结果到`test_voting_metrics.csv`
- 保存`epoch_ids`到`test_epoch_ids.npy`

## 输出指标

投票评估结果包含以下指标：

- `video_accuracy`: 视频级别准确率（考虑平票的0.5分规则）
- `video_bacc`: 视频级别平衡准确率
- `num_videos`: 评估的视频数量
- `subject_X_accuracy`: 每个subject的准确率
- `avg_subject_accuracy`: 平均subject准确率

## 使用方法

训练模型后，测试集的投票评估会自动执行，结果保存在：
- `{log_dir}/test_voting_metrics.csv` - 投票评估指标
- `{log_dir}/test_epoch_ids.npy` - epoch_id列表（用于调试）

## 测试建议

1. 使用少量数据验证`epoch_id`提取是否正确
2. 验证视频分组是否正确（同一视频的chunks应该被分到一组）
3. 验证投票逻辑是否正确（特别是平票情况）
4. 验证subject分组和准确率计算是否正确

