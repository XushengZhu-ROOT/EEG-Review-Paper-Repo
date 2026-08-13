"""
导出测试集上每个 epoch_id 的预测结果，用于 McNemar 配对检验。
输出格式: epoch_id\tpred\ttrue\tcorrect (tab 分隔)
"""
import argparse
import os
import glob
import yaml
import torch
import numpy as np
from tqdm import tqdm

from datasets.motortask_dataset import CustomDataset
from torch.utils.data import DataLoader
from models.model_for_motortask import Model


def load_params_from_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    class Params:
        pass
    params = Params()
    for k, v in cfg.items():
        setattr(params, k, v)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='模型目录，含 config.yaml 和 best_model_*.pth')
    parser.add_argument('--datasets_dir', type=str, default=None,
                        help='数据目录 (train/val/test)，默认用 config 中的 datasets_dir')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径，默认 model_dir/test_predictions.txt')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    params = load_params_from_config(config_path)

    datasets_dir = args.datasets_dir or params.datasets_dir
    if not os.path.isabs(datasets_dir):
        datasets_dir = os.path.normpath(datasets_dir)
    test_dir = os.path.join(datasets_dir, 'test')
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    # 查找 best model
    pth_files = glob.glob(os.path.join(args.model_dir, 'best_model_*.pth'))
    if not pth_files:
        raise FileNotFoundError(f"No best_model_*.pth in {args.model_dir}")
    model_path = sorted(pth_files)[-1]
    print(f"Load model: {model_path}")

    # 构建模型（不加载预训练 backbone，直接用微调后的完整权重）
    params.cuda = args.cuda
    params.use_pretrained_weights = False
    model = Model(params)
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 加载 test 数据
    test_set = CustomDataset(
        datasets_dir, mode='test',
        channel_size=params.channel_size,
        window_size=params.window_size
    )
    loader = DataLoader(
        test_set,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=test_set.collate,
        num_workers=0,
    )

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            # collate 现在返回 (x, y, epoch_ids, sample_ids) 四元组；这里不需要 sample_ids
            x, y, epoch_ids = batch[0], batch[1], batch[2]
            x = x.to(device)
            pred = model(x)
            pred_cls = torch.argmax(pred, dim=-1).cpu().numpy()
            y_np = y.numpy()
            for i, eid in enumerate(epoch_ids):
                p, t = int(pred_cls[i]), int(y_np[i])
                correct = 1 if p == t else 0
                results.append((eid, p, t, correct))

    output_path = args.output or os.path.join(args.model_dir, 'test_predictions.txt')
    with open(output_path, 'w') as f:
        f.write('epoch_id\tpred\ttrue\tcorrect\n')
        for eid, p, t, c in results:
            f.write(f'{eid}\t{p}\t{t}\t{c}\n')
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == '__main__':
    main()
