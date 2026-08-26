#!/usr/bin/env python3
"""
[Sleep] cbramod 专用：加载一个已经训练好的 Sleep checkpoint（不重新训练），
对 val/test 集各做一次干净推理，保存：
  sleep_{model}_val.npz / sleep_{model}_test.npz
    -- sample_id(=epoch_id，已排序) / y_true / y_pred / y_prob(N,5) / subject_id / epoch_index
  sleep_{model}.json
    -- 模型名/任务/lr/wd/bs/best_epoch/val_bacc/test_bacc/数据目录/checkpoint路径/
       val与test各类样本数

跟 labram/biot/sttransformer/eegpt 那几个 Sleep 训练脚本存的是同一套 npz/json schema，
唯一的区别是这里不训练，直接吃一个已经存在的 checkpoint 文件。

lr/weight_decay/batch_size/best_epoch/recorded_val_bacc/recorded_test_bacc 会尝试从
--checkpoint_path 的目录名/文件名自动解析（cbramod 自己的训练脚本本来就用
"hpo_exp{N}_lr{lr}_wd{wd}_bs{bs}-{classifier}-{freeze_type}/best_model_epoch{E}_valBacc{v}_testBacc{v}.pth"
这个命名约定存 checkpoint），解析不出来的字段用 --lr/--weight_decay/--batch_size 等
CLI 参数覆盖；两边都没有就在 sidecar json 里老实存 null，不瞎猜。

用法：
  python3 infer_sleep_checkpoint.py \\
      --checkpoint_path /path/to/hpo_exp4_lr0.0001_wd0.00002_bs64-all_patch_reps-all/best_model_epoch28_valBacc0.77046_testBacc0.77713.pth \\
      --data_root ./cbramod_sleep_data
"""
import argparse
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from datasets.sleep_dataset import CustomDataset
from models.model_for_sleep import Model

_DIRNAME_RE = re.compile(
    r'hpo_exp(?P<exp>\d+)_lr(?P<lr>[0-9.eE+-]+)_wd(?P<wd>[0-9.eE+-]+)_bs(?P<bs>\d+)-(?P<classifier>.+)-(?P<freeze_type>[^-]+)$'
)
_FILENAME_RE = re.compile(
    r'best_model_epoch(?P<epoch>\d+)_valBacc(?P<val_bacc>[0-9.]+)_testBacc(?P<test_bacc>[0-9.]+)\.pth$'
)
_SAMPLE_ID_RE = re.compile(r'^sub(\d+)_ep(\d+)$')


def _parse_checkpoint_path(checkpoint_path):
    """从 checkpoint 路径的目录名/文件名解析训练时的超参数和记录的 epoch/bacc。
    解析不出来的字段返回 None，不抛异常（调用方会用 CLI 参数兜底）。"""
    parsed = {
        'lr': None, 'weight_decay': None, 'batch_size': None, 'classifier': None,
        'freeze_type': None, 'epoch': None, 'recorded_val_bacc': None, 'recorded_test_bacc': None,
    }
    dirname = os.path.basename(os.path.dirname(checkpoint_path))
    m = _DIRNAME_RE.search(dirname)
    if m:
        parsed['lr'] = float(m.group('lr'))
        parsed['weight_decay'] = float(m.group('wd'))
        parsed['batch_size'] = int(m.group('bs'))
        parsed['classifier'] = m.group('classifier')
        parsed['freeze_type'] = m.group('freeze_type')
    else:
        print(f"[warn] could not parse hyperparameters from checkpoint dir name: {dirname!r}")

    filename = os.path.basename(checkpoint_path)
    m = _FILENAME_RE.search(filename)
    if m:
        parsed['epoch'] = int(m.group('epoch'))
        parsed['recorded_val_bacc'] = float(m.group('val_bacc'))
        parsed['recorded_test_bacc'] = float(m.group('test_bacc'))
    else:
        print(f"[warn] could not parse epoch/val_bacc/test_bacc from checkpoint file name: {filename!r}")

    return parsed


def _collate_with_sample_id(batch):
    x = np.array([b[0] for b in batch])
    y = np.array([b[1] for b in batch])
    sample_ids = [b[2] for b in batch]
    return torch.from_numpy(x).float(), torch.from_numpy(y).long(), sample_ids


def run_split(model, device, data_root, split_name, channel_size, window_size, batch_size, n_classes):
    dataset = CustomDataset(
        data_root, mode=split_name, channel_size=channel_size, window_size=window_size,
        return_sample_id=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_with_sample_id)

    sample_ids, y_true, y_pred, y_prob, subject_ids, epoch_indices = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, batch_sample_ids in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = torch.argmax(logits, dim=-1).cpu()
            for i, sid in enumerate(batch_sample_ids):
                m = _SAMPLE_ID_RE.match(sid)
                if not m:
                    raise ValueError(f"Cannot parse subject_id/epoch_index from sample_id: {sid!r}")
                sample_ids.append(sid)
                y_true.append(int(y[i].item()))
                y_pred.append(int(preds[i].item()))
                y_prob.append(probs[i].numpy())
                subject_ids.append(int(m.group(1)))
                epoch_indices.append(int(m.group(2)))

    if len(sample_ids) == 0:
        raise RuntimeError(f"run_split: no samples collected for split={split_name!r} under {data_root!r}.")

    sample_ids_arr = np.array(sample_ids)
    order = np.argsort(sample_ids_arr)
    sample_ids_arr = sample_ids_arr[order]
    y_true_arr = np.array(y_true, dtype=np.int64)[order]
    y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
    y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
    subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]
    epoch_index_arr = np.array(epoch_indices, dtype=np.int64)[order]

    bacc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    class_counts = np.bincount(y_true_arr, minlength=n_classes).tolist()
    return {
        'sample_id': sample_ids_arr, 'y_true': y_true_arr, 'y_pred': y_pred_arr,
        'y_prob': y_prob_arr, 'subject_id': subject_id_arr, 'epoch_index': epoch_index_arr,
    }, bacc, class_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                         help='已经训练好的 Sleep checkpoint (.pth，plain state_dict)，必须显式指定，不提供默认值')
    parser.add_argument('--data_root', type=str, default='./cbramod_sleep_data',
                         help='真实 Sleep 数据目录（含 train/val/test 子目录）')
    parser.add_argument('--channel_size', type=int, default=6)
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--num_of_classes', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=None,
                         help='推理时的 batch size；不传则用从 checkpoint 路径解析出来的训练 batch_size')
    parser.add_argument('--lr', type=float, default=None, help='仅用于记录进 sidecar json；不传则尝试从路径解析')
    parser.add_argument('--weight_decay', type=float, default=None, help='仅用于记录进 sidecar json；不传则尝试从路径解析')
    parser.add_argument('--classifier', type=str, default=None, help='不传则尝试从路径解析，解析不出来时必填')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--task_name', type=str, default='sleep')
    parser.add_argument('--model_name', type=str, default='cbramod')
    parser.add_argument('--fold_results_dir', type=str, default='./fold_results_cbramod_sleep')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"--checkpoint_path does not exist: {args.checkpoint_path}")

    parsed = _parse_checkpoint_path(args.checkpoint_path)
    lr = args.lr if args.lr is not None else parsed['lr']
    weight_decay = args.weight_decay if args.weight_decay is not None else parsed['weight_decay']
    batch_size = args.batch_size if args.batch_size is not None else (parsed['batch_size'] or 64)
    classifier = args.classifier if args.classifier is not None else parsed['classifier']
    if classifier is None:
        raise ValueError(
            "--classifier could not be parsed from --checkpoint_path and was not given explicitly; "
            "pass --classifier (e.g. all_patch_reps) so the model architecture can be rebuilt correctly."
        )

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    model_params = SimpleNamespace(
        use_pretrained_weights=False,  # 马上就要整个用 checkpoint 覆盖，不需要再加载 foundation 权重
        foundation_dir=None,
        cuda=args.cuda,
        num_of_classes=args.num_of_classes,
        classifier=classifier,
        channel_size=args.channel_size,
        window_size=args.window_size,
        dropout=args.dropout,
    )
    model = Model(model_params)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    os.makedirs(args.fold_results_dir, exist_ok=True)
    task = args.task_name
    model_name = args.model_name

    val_data, val_bacc, val_class_counts = run_split(
        model, device, args.data_root, 'val', args.channel_size, args.window_size, batch_size, args.num_of_classes)
    test_data, test_bacc, test_class_counts = run_split(
        model, device, args.data_root, 'test', args.channel_size, args.window_size, batch_size, args.num_of_classes)

    val_npz_path = os.path.join(args.fold_results_dir, f"{task}_{model_name}_val.npz")
    test_npz_path = os.path.join(args.fold_results_dir, f"{task}_{model_name}_test.npz")
    np.savez(val_npz_path, **val_data)
    np.savez(test_npz_path, **test_data)
    for path, data in ((val_npz_path, val_data), (test_npz_path, test_data)):
        if not os.path.exists(path):
            raise RuntimeError(f"failed to write {path}")
        reload = np.load(path)
        for key in ('sample_id', 'y_true', 'y_pred', 'y_prob', 'subject_id', 'epoch_index'):
            if key not in reload or len(reload[key]) != len(data['sample_id']):
                raise RuntimeError(f"{path}: key {key!r} missing or length mismatch after write")
    print(f"Saved {val_npz_path} (balanced_accuracy={val_bacc:.5f})")
    print(f"Saved {test_npz_path} (balanced_accuracy={test_bacc:.5f})")

    if parsed['recorded_val_bacc'] is not None:
        match = abs(parsed['recorded_val_bacc'] - val_bacc) < 1e-3
        print(f"[check] checkpoint filename recorded val_bacc={parsed['recorded_val_bacc']:.5f} vs "
              f"recomputed {val_bacc:.5f} -> {'OK, matches' if match else 'MISMATCH -- check --data_root is the same split used to train this checkpoint'}")

    meta = {
        'model_name': model_name,
        'task': task,
        'dataset': 'Sleep',
        'split_mode': 'pooled_random_epoch',  # 见 preprocess_sleep.py：全体受试者按 epoch 分层随机切分，不是 LOSO
        'dataset_path': args.data_root,
        'best_epoch': parsed['epoch'],
        'val_balanced_accuracy': val_bacc,
        'test_balanced_accuracy': test_bacc,
        'recorded_val_balanced_accuracy': parsed['recorded_val_bacc'],
        'recorded_test_balanced_accuracy': parsed['recorded_test_bacc'],
        'val_class_counts': val_class_counts,
        'test_class_counts': test_class_counts,
        'checkpoint_path': args.checkpoint_path,
        'val_npz_path': val_npz_path,
        'test_npz_path': test_npz_path,
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'hyperparams': {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'classifier': classifier,
            'freeze_type': parsed['freeze_type'],
            'channel_size': args.channel_size,
            'window_size': args.window_size,
        },
    }
    json_path = os.path.join(args.fold_results_dir, f"{task}_{model_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if not os.path.exists(json_path):
        raise RuntimeError(f"failed to write {json_path}")
    with open(json_path) as f:
        json.load(f)
    print(f"Saved {json_path}")
    print(f"  best_epoch={parsed['epoch']} val_bacc={val_bacc:.5f} test_bacc={test_bacc:.5f}")


if __name__ == '__main__':
    main()
