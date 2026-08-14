"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import os
import sys
import time
import json
import yaml
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from pathlib import Path
import tiktoken
from utils import prepare_KaggleERN_dataset, prepare_STRESS_dataset, prepare_SEED7_dataset, prepare_motor_dataset, prepare_sleep_dataset, cosine_scheduler, get_metrics
from utils import prepare_motor_dataset_loso
from torch.utils.data.dataset import ConcatDataset


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None

def make_json_serializable(obj):
    """递归把 numpy 标量/数组等转成可 json.dumps 的 Python 原生类型。"""
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj

def _print_confusion_matrices(results, dataset_info, split_name):
    """把任意 *confusion_matrix* 字段按行打印，避免一行挤爆。"""
    label_names = dataset_info.get('label_names', None)
    matrix_keys = [k for k in results.keys() if 'confusion_matrix' in k and not k.endswith('_labels')]
    for mk in sorted(matrix_keys):
        labels_key = mk + "_labels" if (mk + "_labels") in results else (mk.replace("confusion_matrix", "confusion_matrix_labels"))
        cm = results.get(mk, None)
        if cm is None:
            continue
        labels = results.get(labels_key, None)
        print(f"Confusion Matrix ({split_name}) - {mk}:")
        if labels is not None:
            if label_names is not None:
                # label_names 是按 class id 顺序的
                try:
                    names = [label_names[i] for i in labels]
                except Exception:
                    names = label_names
                print(f"Labels: {names}")
            else:
                print(f"Labels: {labels}")
        for row in cm:
            print(row)


def compute_class_token_ids(loader, num_classes):
    """从数据集自身的 prompt / answer 文本张量推导每个类别的“答案 token id”及答案位置 k。

    找出各类别 answer 文本第一个互不相同的 token 位置 k，该位置的 token 即类别 token。
    评估时：喂入各类别的“共同前缀” answer_text[:k]（即读完 prompt 后应生成第一个答案 token 之前的上下文），
    读取最后位置的 logits，对这些类别 token 取 softmax 即得到类别概率。

    这样对 GPT-2 BPE 是否把 '(' 与字母合并都成立：
      - 不合并: k == len(prompt)，前缀就是完整 prompt，预测的是字母 token；
      - 合并:   k == len(prompt)-1，前缀是去掉尾部 '(' 的 prompt，预测的是 '(A' 这类合并 token。
    两种情况都与训练时的监督位置一致（训练用 prompt_len-1 对齐）。

    自检（fail loud）：
      - 必须存在互不相同的位置 k；
      - loader.prompt 的前 k 个 token 必须等于共同前缀（即前缀与 prompt 一致）；
      - 1 <= k <= len(prompt)；
      - 各类别 token 两两不同。
    返回 (class_token_ids, k)。
    """
    if not hasattr(loader, 'prompt') or not hasattr(loader, 'text'):
        raise RuntimeError("loader 没有 prompt / text，无法推导 class token id（需 is_instruct=True）")

    prompt = loader.prompt.cpu()
    p = int(prompt.size(0))
    texts = [loader.text[c].cpu() for c in range(num_classes)]
    min_len = min(int(t.size(0)) for t in texts)

    k = None
    for i in range(min_len):
        toks_i = set(int(t[i].item()) for t in texts)
        if len(toks_i) > 1:
            k = i
            break
    if k is None:
        raise RuntimeError("各类别 answer 文本没有任何互不相同的 token，无法区分类别")
    if not (1 <= k <= p):
        raise RuntimeError(f"答案位置 k={k} 不在 [1, prompt_len={p}] 内，假设不成立")
    # 共同前缀应与 prompt 的前 k 个 token 一致
    if not torch.equal(prompt[:k], texts[0][:k]):
        raise RuntimeError("answer 文本的共同前缀与 prompt 不一致，假设不成立")

    ids = [int(t[k].item()) for t in texts]
    if len(set(ids)) != num_classes:
        raise RuntimeError(f"类别 token id 不是两两不同: {ids}")
    print(f"class_token_ids (num_classes={num_classes}): {ids}, answer_pos k={k}, prompt_len p={p}")
    return ids, k


def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = 'float32'

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_instruct_datasets(args, downstream_dataset: str, eeg_max_len=-1, text_max_len=-1, fold=None):
        dataset_info = {'name': downstream_dataset}
        dataset_info['loso_meta'] = None
        if downstream_dataset == 'KaggleERN':
            dataset_train, dataset_test, dataset_val = prepare_KaggleERN_dataset(Path(args.dataset_dir), chan_size=args.chan_size, is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            #TODO: 待修改
            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 9
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}
        elif downstream_dataset == 'Stress':
            dataset_train, dataset_test, dataset_val = prepare_STRESS_dataset(Path(args.dataset_dir), chan_size=args.chan_size, is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 9
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}
        elif downstream_dataset == 'SEED7':
            dataset_train, dataset_test, dataset_val = prepare_SEED7_dataset(Path(args.dataset_dir), chan_size=args.chan_size, is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 6
            dataset_info['result_idx'] = 22  # Answer: (X) is at word index 22 (0-indexed) after "Answer:"
            # Generated text format: "Question: ... Options: (A) ... (F) ... Answer: (X) <|endoftext|>)"
            # Answer "(X)" is at word index 22 (0-indexed, after splitting by space)
            # Verified from debug output: "Answer:" is at index 21, answer label is at index 22
            # Note: neutral class is excluded
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5}
            dataset_info['label_names'] = ['happy', 'sad', 'disgust', 'fear', 'surprise', 'anger']
            
            # Debug: Print prompt format to verify result_idx calculation
            # Commented out for production use - uncomment for debugging
            # if master_process:
            #     # Calculate result_idx based on prompt format
            #     prompt_text = 'Question: Which emotion does this EEG segment express? Options: (A) happy. (B) sad. (C) disgust. (D) fear. (E) surprise. (F) anger. Answer: ('
            #     words = prompt_text.split(' ')
            #     # The answer format is "Answer: (X)" where X is the label
            #     # In the generated text: "Answer: (C) <|endoftext|>", the answer (C) is at index 23 (0-indexed)
            #     # Prompt has 23 words, so answer will be at index 23 in generated text
            #     expected_result_idx = 23
            #     
            #     print(f"\n{'='*80}")
            #     print("DEBUG: SEED7 Dataset Configuration - Prompt Format Analysis")
            #     print(f"{'='*80}")
            #     print(f"Prompt text: {prompt_text}")
            #     print(f"Prompt words count: {len(words)}")
            #     print(f"Configured result_idx: {dataset_info['result_idx']} (will be set dynamically during evaluation)")
            #     print(f"Expected answer position in generated text: index {expected_result_idx} (Answer: (X))")
            #     print(f"Note: result_idx will be automatically determined from first successful prediction during evaluation.")
            #     print(f"{'='*80}\n")
        elif downstream_dataset == 'MOTOR':
            if fold is not None:
                # leave-one-subject-out: 忽略原始 train/val/test 划分，按被试重新分折
                dataset_train, dataset_test, dataset_val, loso_meta = prepare_motor_dataset_loso(
                    Path(args.dataset_dir), fold=fold, is_instruct=True,
                    eeg_max_len=eeg_max_len, text_max_len=text_max_len, n_folds=args.n_folds)
                dataset_info['loso_meta'] = loso_meta
            else:
                dataset_train, dataset_test, dataset_val = prepare_motor_dataset(Path(args.dataset_dir), is_instruct=True,
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 6
            dataset_info['result_idx'] = 25  # Answer: (A) or (B) etc. position in the generated text (0-indexed)
            # Generated text format: "Question: ... Options: (A) ... (F) ... Answer: (X) <|endoftext|>)"
            # Answer "(X)" is at word index 25 (after splitting by space, "Answer:" is at 24, label is at 25)
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5}
            dataset_info['label_names'] = ['Label0', 'Walk', '8', 'Horizontal', 'Vertical', 'Pick']
        elif downstream_dataset == 'SLEEP':
            dataset_train, dataset_test, dataset_val = prepare_sleep_dataset(Path(args.dataset_dir), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 5
            dataset_info['result_idx'] = 27  # Answer: (A) or (B) etc. position in the generated text (0-indexed)
            # Generated text format: "Question: ... Options: (A) ... (E) ... Answer: (X) <|endoftext|>)"
            # Answer "(X)" is at word index 22 (after splitting by space, "Answer:" is at 21, label is at 22)
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4}
            dataset_info['label_names'] = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

        dataset_info['dataset_train'] = dataset_train
        dataset_info['dataset_val'] = dataset_val
        dataset_info['dataset_test'] = dataset_test

        # LOSO 评估：数据集会额外返回 sample_id / subject_id，并用答案位置 logits 计算 softmax 概率
        dataset_info['has_sample_id'] = (fold is not None)
        if fold is not None:
            _cti, _apos = compute_class_token_ids(dataset_test, dataset_info['num_classes'])
            dataset_info['class_token_ids'] = _cti
            dataset_info['answer_pos'] = _apos

        if ddp:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
            )
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
            )
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
            )
        else:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        dataset_info['data_loader_train'] = data_loader_train
        dataset_info['data_loader_val'] = data_loader_val
        dataset_info['data_loader_test'] = data_loader_test
        return dataset_info


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/instruction-B')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        config_path = os.path.join(args.out_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(vars(args), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"Hyperparameters saved to {config_path}")

    # local training logs
    log_file_path = os.path.join(args.out_dir, "training_log.jsonl")

    # text data loader
    data_dir = os.path.join(args.text_data_dir)
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    concat_datasets = True
    all_datasets = []
    
    # for name in ['TUAB', 'TUEV', 'SEED', 'HMC', 'Workload', 'TUSL', 'Stress']:
        # all_datasets.append(get_instruct_datasets(args, name, eeg_max_len=276, text_max_len=80))
    dataset_dir_lower = args.dataset_dir.lower()
    if 'stress' in dataset_dir_lower:
        name = 'Stress'
    elif 'kaggleern' in dataset_dir_lower:
        name = 'KaggleERN'
    elif 'seed' in dataset_dir_lower or 'seed_data' in dataset_dir_lower:
        name = 'SEED7'
    elif 'motor' in dataset_dir_lower or 'motor_data' in dataset_dir_lower:
        name = 'MOTOR'
    elif 'sleep' in dataset_dir_lower or 'sleep_data' in dataset_dir_lower:
        name = 'SLEEP'
    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset_dir}\n"
            f"Path must contain: ['stress', 'kaggleern', 'seed', 'motor', 'sleep']"
        )
    all_datasets.append(get_instruct_datasets(args, name, eeg_max_len=248, text_max_len=80, fold=args.fold))
        
    if concat_datasets:
        merge_datasets = ConcatDataset([dataset_info['dataset_train'] for dataset_info in all_datasets])
        if ddp:
            sampler_merge = torch.utils.data.DistributedSampler(
                merge_datasets, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets, sampler=sampler_merge,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True
            )
        else:
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
            
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    tokenizer_ckpt_path = os.path.join(args.out_dir, args.tokenizer_path)

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'pretrained'
    # model init
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                    bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
    if init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='gpt2')
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    elif init_from == 'gpt':
        print(f"Initializing from tokenizer weights: {init_from}")
        # initialize from EEGPT weights
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                        bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, tokenizer_ckpt_path, init_from='gpt2')
        start_epoch = 0
    elif init_from == 'pretrained':
        print(f"Initializing training from {args.NeuroLM_path}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.NeuroLM_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='scratch')
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_epoch = 0

    model.to(device)
    if master_process:
        model_structure_path = os.path.join(args.out_dir, "model_structure.txt")
        with open(model_structure_path, 'w') as f:
            f.write(str(model))
        print(f"Model structure saved to {model_structure_path}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project, name=args.wandb_runname, dir=os.path.join(args.out_dir), resume=True)


    num_training_steps_per_epoch = sum([len(dataset['dataset_train']) for dataset in all_datasets]) // args.eeg_batch_size // ddp_world_size
    # Ensure at least 1 step per epoch to avoid division issues
    num_training_steps_per_epoch = max(1, num_training_steps_per_epoch)
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=int(args.warmup_ratio * num_training_steps_per_epoch * args.epochs)
    )
    
    # Log schedule info for debugging - commented out for production use
    # if master_process:
    #     print(f"Learning rate schedule: {len(lr_schedule_values)} steps total (epochs={args.epochs}, steps_per_epoch={num_training_steps_per_epoch})")
    #     print(f"  Expected total iterations: {args.epochs * num_training_steps_per_epoch}")
    #     print(f"  LR schedule array size: {len(lr_schedule_values)}")

    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    
    # training loop
    datasets = [{'data_loader_train': data_loader_merge}] if concat_datasets else all_datasets
    X_text2, Y_text2 = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    if args.eval_only:
        start_epoch = 0

    # ===== LOSO 状态：验证集选最佳 epoch + 训练计时 + 显存峰值 =====
    loso_enabled = args.fold is not None
    best_val_bacc = -1.0
    best_epoch = -1
    best_test_payload = None
    train_time_sec = 0.0
    if loso_enabled and device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        for dataset_info in datasets:
            if args.eval_only:
                break
            for step, (batch) in enumerate(dataset_info['data_loader_train']):
                # determine and set the learning rate for this iteration
                # Add boundary check to handle cases where iter_num might exceed schedule length
                # (e.g., due to drop_last=False or data loader behavior differences)
                if args.decay_lr:
                    if iter_num < len(lr_schedule_values):
                        lr = lr_schedule_values[iter_num]
                    else:
                        # Use the last learning rate value if we exceed the schedule
                        lr = lr_schedule_values[-1]
                        if master_process and iter_num == len(lr_schedule_values):
                            print(f"Warning: iter_num ({iter_num}) exceeded LR schedule length ({len(lr_schedule_values)}). Using final LR: {lr}")
                else:
                    lr = args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                X_eeg, X_text, Y_text, input_chans, input_time, input_mask, gpt_mask = batch
                
                X_eeg = X_eeg.float().to(device, non_blocking=True)
                X_text = X_text.to(device, non_blocking=True)
                Y_text = Y_text.to(device, non_blocking=True)
                input_chans = input_chans.to(device, non_blocking=True)
                input_time = input_time.to(device, non_blocking=True)
                gpt_mask = gpt_mask.to(device, non_blocking=True)
                if input_mask is not None:
                    input_mask = input_mask.to(device, non_blocking=True)

                Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device, non_blocking=True)

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

                with ctx:
                    loss1, log1, logits = model(X_eeg, Y_eeg, X_text, Y_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask)
                    loss2, log2, _ = model(None, None, X_text2, Y_text2)
                    
                    model.train()

                    loss = (loss1 + loss2) / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # clip the gradient
                    if args.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    # step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                
                X_text2, Y_text2 = get_batch('train')

                # evaluate the loss on train/val sets and write checkpoints
                if (iter_num + 1) % args.log_interval == 0 and master_process:
                    print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train total loss {log1['train/loss'] + log2['train/loss']:.4f}, instruction loss {log1['train/loss']:.4f}, text loss {log2['train/loss']:.4f}")
                    log_data = {
                        "type": "train",
                        "epoch": epoch,
                        "iter_num": iter_num,
                        "step": step + 1,
                        "train/total_loss": log1['train/loss'] + log2['train/loss'],
                        "train/instruction_loss": log1['train/loss'],
                        "train/text_loss": log2['train/loss'],
                        "train/instruction_accuracy": log1['train/accuracy'],
                        "train/text_accuracy": log2['train/accuracy'],
                        "lr": lr
                    }
                    if args.wandb_log:
                        wandb.log(log_data)

                    with open(log_file_path, 'a') as f:
                        f.write(json.dumps(log_data) + '\n')

                if iter_num == 0 and args.eval_only:
                    break

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                iter_num += 1
                local_iter_num += 1
        
        # Checkpoint saving disabled - no longer saving checkpoints
        # if master_process and (not args.eval_only):
        #     checkpoint = {
        #         'model': raw_model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'model_args': model_args,
        #         'iter_num': iter_num,
        #         'epoch': epoch
        #     }
        #     print(f"saving checkpoint to {checkpoint_out_dir}")
        #     torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        #     if (epoch + 1) % args.save_ckpt_freq == 0:
        #         print(f"saving checkpoint to {checkpoint_out_dir}")
        #         torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
        
        # 累计训练耗时（不含评估）
        train_time_sec += time.time() - epoch_train_start

        # ===== LOSO：用验证集 balanced_accuracy 选最佳 epoch；测试集结果留到折末保存 =====
        if loso_enabled:
            for dataset_info in all_datasets:
                val_payload = evaluate_probs(raw_model, dataset_info, dataset_info['data_loader_val'])
                test_payload = evaluate_probs(raw_model, dataset_info, dataset_info['data_loader_test'])
                val_bacc = float(balanced_accuracy_score(val_payload['y_true'], val_payload['y_pred']))
                test_bacc = float(balanced_accuracy_score(test_payload['y_true'], test_payload['y_pred']))
                print(f"[fold {args.fold}] epoch {epoch}: val_bacc={val_bacc:.4f} test_bacc={test_bacc:.4f}")
                if master_process and args.wandb_log:
                    wandb.log({
                        f'val_{dataset_info["name"]}/balanced_accuracy': val_bacc,
                        f'test_{dataset_info["name"]}/balanced_accuracy': test_bacc,
                        'epoch': epoch,
                    })
                # 验证集更优则记录该 epoch 的测试集预测
                if val_bacc > best_val_bacc:
                    best_val_bacc = val_bacc
                    best_epoch = epoch
                    best_test_payload = test_payload
            if args.eval_only:
                break
            continue

        # validation and test
        for dataset_info in all_datasets:
            print('Dataset:', dataset_info['name'])
            results_val = evaluate(raw_model, dataset_info, dataset_info['data_loader_val'], decode)
            print('=' * 10)
            print('Eval:')
            for metric in results_val.keys():
                # 混淆矩阵相关字段单独换行打印
                if 'confusion_matrix' not in metric:
                    print(metric + ':', results_val[metric])
            _print_confusion_matrices(results_val, dataset_info, "Val")
            results_test = evaluate(raw_model, dataset_info, dataset_info['data_loader_test'], decode)
            print('=' * 10)
            print('Test:')
            for metric in results_test.keys():
                if 'confusion_matrix' not in metric:
                    print(metric + ':', results_test[metric])
            _print_confusion_matrices(results_test, dataset_info, "Test")
            print('=' * 10)
            if master_process:
                local_log_data = {
                    "type": "eval",
                    "epoch": epoch,
                    "dataset": dataset_info['name'],
                    "val_results": results_val,
                    "test_results": results_test
                }
                with open(log_file_path, 'a') as f:
                    f.write(json.dumps(make_json_serializable(local_log_data)) + '\n')
                    
            if args.wandb_log and master_process:
                wandb_log_data = {}
                # 分别处理val和test的指标，只记录实际存在的
                for metric in results_val.keys():
                    # 跳过混淆矩阵和标签（它们已经在日志文件中记录，wandb不支持直接记录二维数组）
                    if 'confusion_matrix' not in metric:
                        wandb_log_data['val_' + dataset_info['name'] + '/' + metric] = results_val[metric]
                for metric in results_test.keys():
                    # 跳过混淆矩阵和标签
                    if 'confusion_matrix' not in metric:
                        wandb_log_data[f'test_' + dataset_info['name'] + '/' + metric] = results_test[metric]
                wandb.log(make_json_serializable(wandb_log_data))
                
                # 如果有混淆矩阵，可以记录预测成功率
                if results_val.get('pred_success_rate') is not None:
                    wandb.log({
                        'val_' + dataset_info['name'] + '/pred_success_rate': results_val['pred_success_rate'],
                        'test_' + dataset_info['name'] + '/pred_success_rate': results_test.get('pred_success_rate', 0.0)
                    })
        if args.eval_only:
            break

    # ===== 折末：保存该折测试集结果（npz + json）。保存失败直接报错退出 =====
    if loso_enabled:
        if best_test_payload is None:
            raise RuntimeError(f"[fold {args.fold}] 没有可保存的测试结果（best_test_payload 为空）")
        peak_gpu_mem_mb = 0.0
        gpu_name = 'cpu'
        if device_type == 'cuda':
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            gpu_name = torch.cuda.get_device_name(device)
        loso_meta = all_datasets[0].get('loso_meta') or {}
        hyperparams = {
            'learning_rate': args.learning_rate,
            'min_lr': args.min_lr,
            'weight_decay': args.weight_decay,
            'beta1': args.beta1,
            'beta2': args.beta2,
            'eeg_batch_size': args.eeg_batch_size,
            'text_batch_size': args.text_batch_size,
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'warmup_ratio': args.warmup_ratio,
            'grad_clip': args.grad_clip,
            'block_size': args.block_size,
            'decay_lr': bool(args.decay_lr),
            'seed': args.seed,
        }
        fold_stats = {
            'test_subject': loso_meta.get('test_subject'),
            'val_subject': loso_meta.get('val_subject'),
            'best_epoch': best_epoch,
            'best_val_bacc': best_val_bacc,
            'hyperparams': hyperparams,
            'train_time_sec': train_time_sec,
            'peak_gpu_mem_mb': peak_gpu_mem_mb,
            'gpu_name': gpu_name,
        }
        if master_process:
            try:
                save_fold_results(args, all_datasets[0]['name'], best_test_payload, fold_stats)
            except Exception as e:
                print(f"[fold {args.fold}] 保存结果失败: {e}", file=sys.stderr)
                if ddp:
                    destroy_process_group()
                raise

    if ddp:
        destroy_process_group()


def get_pred(pred_string, dataset_info, debug=False):
    """
    从生成的文本中提取预测标签
    
    Returns:
        pred: 预测的标签索引，如果失败返回 -1
        debug_info: 如果 debug=True，返回调试信息字典，包含找到的标签和位置
    """
    debug_info = {'method': None, 'found_label': None, 'label_position': None, 'answer_part': None}
    
    if dataset_info['name'] == 'zuco':
        pred = pred_string[17:].split('<|endoftext|>')[0]
        debug_info['method'] = 'zuco_special'
    else:
        pred = -1
        try:
            # 首先尝试按 result_idx 位置解析（如果已设置）
            words = pred_string.split(' ')
            if dataset_info.get('result_idx') is not None and len(words) > dataset_info['result_idx']:
                pred_word = words[dataset_info['result_idx']]
                debug_info['method'] = 'result_idx'
                debug_info['label_position'] = dataset_info['result_idx']
                debug_info['original_word'] = pred_word
                
                # 鲁棒性处理：从可能包含额外字符的字符串中提取有效标签
                # 例如：从 "(C))" 或 "(F))" 中提取 "(C)" 或 "(F)"
                extracted_label = None
                
                # 方法1: 如果直接匹配，使用它
                if pred_word in dataset_info['label_dic']:
                    extracted_label = pred_word
                # 方法2: 尝试提取前3个字符（标准格式是 "(X)"）
                elif pred_word.startswith('(') and len(pred_word) >= 3:
                    candidate = pred_word[:3]
                    if candidate in dataset_info['label_dic']:
                        extracted_label = candidate
                # 方法3: 在字符串中搜索所有可能的标签
                else:
                    for label_key in dataset_info['label_dic'].keys():
                        if label_key in pred_word:
                            extracted_label = label_key
                            break
                
                if extracted_label and extracted_label in dataset_info['label_dic']:
                    pred = dataset_info['label_dic'][extracted_label]
                    debug_info['found_label'] = extracted_label
                    if extracted_label != pred_word:
                        debug_info['original_word'] = pred_word
                        debug_info['extracted_label'] = extracted_label
                else:
                    # result_idx位置没有找到有效标签，继续使用fallback方法
                    raise KeyError(f"Could not extract valid label from '{pred_word}'")
            else:
                # 如果文本太短，尝试在 "Answer:" 之后搜索标签（避免匹配到 Options 部分的标签）
                answer_part = pred_string
                if 'Answer:' in pred_string:
                    answer_part = pred_string.split('Answer:')[-1]
                debug_info['method'] = 'search_in_answer'
                debug_info['answer_part'] = answer_part
                
                for label_key in dataset_info['label_dic'].keys():
                    if label_key in answer_part:
                        # 找到标签在 answer_part 中的位置
                        label_pos = answer_part.find(label_key)
                        # 计算在整个文本中的位置（相对于 "Answer:" 之后）
                        words_before_answer = len(pred_string.split('Answer:')[0].split(' '))
                        words_in_answer_before_label = len(answer_part[:label_pos].split(' '))
                        actual_position = words_before_answer + words_in_answer_before_label
                        
                        debug_info['found_label'] = label_key
                        debug_info['label_position'] = actual_position
                        pred = dataset_info['label_dic'][label_key]
                        break
        except (KeyError, IndexError) as e:
            # 如果按位置解析失败，尝试在 "Answer:" 之后搜索标签
            try:
                answer_part = pred_string
                if 'Answer:' in pred_string:
                    answer_part = pred_string.split('Answer:')[-1]
                debug_info['method'] = 'search_in_answer_fallback'
                debug_info['answer_part'] = answer_part
                
                for label_key in dataset_info['label_dic'].keys():
                    if label_key in answer_part:
                        # 找到标签在 answer_part 中的位置
                        label_pos = answer_part.find(label_key)
                        words_before_answer = len(pred_string.split('Answer:')[0].split(' '))
                        words_in_answer_before_label = len(answer_part[:label_pos].split(' '))
                        actual_position = words_before_answer + words_in_answer_before_label
                        
                        debug_info['found_label'] = label_key
                        debug_info['label_position'] = actual_position
                        pred = dataset_info['label_dic'][label_key]
                        break
            except:
                pred = -1
                debug_info['method'] = 'failed'
    
    if debug:
        return pred, debug_info
    return pred

def extract_video_index(epoch_id):
    """从 epoch_id 中提取 video_index
    例如: 'subject_1_video_index_3_chunk001' -> 3
    """
    import re
    match = re.search(r'video_index_(\d+)_chunk', epoch_id)
    if match:
        return int(match.group(1))
    return None

def extract_subject_id(epoch_id):
    """从 epoch_id 中提取 subject_id
    例如: 'subject_1_video_index_3_chunk001' -> 1
    """
    import re
    match = re.search(r'subject_(\d+)_', epoch_id)
    if match:
        return int(match.group(1))
    return None

def vote_predictions(chunk_preds, chunk_targets, epoch_ids):
    """
    对同一video的所有chunks进行投票，得到video级别的预测和真实标签
    
    Args:
        chunk_preds: list of chunk预测标签 (可能包含-1表示预测失败)
        chunk_targets: list of chunk真实标签
        epoch_ids: list of chunk的epoch_id
        
    Returns:
        video_results: dict {
            'video_index': video_index,
            'subject_id': subject_id,
            'video_pred': 投票后的预测标签,
            'video_target': 真实标签 (应该所有chunks相同),
            'vote_score': 投票得分 (1.0=正确, 0.5=平票且真实标签在候选中, 0.0=错误),
            'chunk_preds': 该video的所有chunk预测,
            'num_chunks': chunk数量
        }
    """
    # 按video_index分组
    video_groups = {}
    for pred, target, epoch_id in zip(chunk_preds, chunk_targets, epoch_ids):
        video_index = extract_video_index(epoch_id)
        subject_id = extract_subject_id(epoch_id)
        
        if video_index is None:
            continue
        
        key = (subject_id, video_index)
        if key not in video_groups:
            video_groups[key] = {
                'preds': [],
                'targets': [],
                'epoch_ids': []
            }
        
        video_groups[key]['preds'].append(pred)
        video_groups[key]['targets'].append(target)
        video_groups[key]['epoch_ids'].append(epoch_id)
    
    # 对每个video进行投票
    video_results = []
    for (subject_id, video_index), group in video_groups.items():
        preds = group['preds']
        targets = group['targets']
        
        # 过滤掉预测失败的chunks (-1)
        valid_preds = [p for p in preds if p != -1]
        
        if len(valid_preds) == 0:
            # 所有chunks都预测失败
            video_pred = -1
            vote_score = 0.0
        else:
            # 多数投票
            from collections import Counter
            pred_counts = Counter(valid_preds)
            max_count = max(pred_counts.values())
            max_labels = [label for label, count in pred_counts.items() if count == max_count]
            
            if len(max_labels) == 1:
                # 没有平票
                video_pred = max_labels[0]
            else:
                # 平票：如果有多个标签得票相同
                video_pred = max_labels  # 保存所有平票标签
        
        # 真实标签（应该所有chunks相同，取第一个）
        video_target = targets[0] if targets else None
        
        # 计算投票得分
        if video_pred == -1:
            vote_score = 0.0
        elif isinstance(video_pred, list):
            # 平票情况
            if video_target in video_pred:
                vote_score = 0.5  # 真实标签在平票候选中
            else:
                vote_score = 0.0  # 真实标签不在平票候选中
        else:
            # 正常投票结果
            if video_pred == video_target:
                vote_score = 1.0
            else:
                vote_score = 0.0
        
        video_results.append({
            'video_index': video_index,
            'subject_id': subject_id,
            'video_pred': video_pred if not isinstance(video_pred, list) else video_pred[0],  # 平票时取第一个作为代表
            'video_pred_all': video_pred if isinstance(video_pred, list) else [video_pred],  # 保存所有可能的预测（用于平票情况）
            'video_target': video_target,
            'vote_score': vote_score,
            'chunk_preds': preds,
            'num_chunks': len(preds),
            'num_valid_chunks': len(valid_preds)
        })
    
    return video_results

@torch.no_grad()
def evaluate(model, dataset_info, dataloader, decode):
    model.eval()
    preds = []
    preds_raw = []  # 保存原始预测标签用于混淆矩阵
    targets = []
    epoch_ids = []  # 保存epoch_id用于投票
    # Debug variables - commented out for production use
    # debug_sample_count = 0
    # max_debug_samples = 5  # 只对前5个样本进行debug输出
    
    # 检查是否是SEED7数据集，需要投票机制
    use_voting = dataset_info.get('name') == 'SEED7'
    
    for batch_idx, (batch) in enumerate(dataloader):
        if use_voting:
            # SEED7: 返回包含epoch_id
            X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask, batch_epoch_ids = batch
        else:
            # 其他数据集: 不包含epoch_id
            X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
            batch_epoch_ids = None
        X_eeg = X_eeg.float().to(device, non_blocking=True)
        X_text = X_text.to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)
        if input_mask is not None:
            input_mask = input_mask.to(device, non_blocking=True)

        with ctx:
            text = model.generate(X_eeg, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
            text = text[:, 1:] # remove [SEP] token
            for i, t in enumerate(text):
                # Convert to list for processing
                token_list = t.tolist()
                
                # Find the first endoftext token (ID 50256) and truncate there
                endoftext_id = 50256
                if endoftext_id in token_list:
                    endoftext_idx = token_list.index(endoftext_id)
                    # Include the endoftext token itself, but nothing after
                    token_list = token_list[:endoftext_idx + 1]
                
                # Decode the cleaned token sequence
                pred_string = decode(token_list)
                
                # Additional text-level cleanup: remove everything after <|endoftext|> (safety check)
                if '<|endoftext|>' in pred_string:
                    pred_string = pred_string.split('<|endoftext|>')[0] + '<|endoftext|>'
                
                # Debug输出：对前几个样本进行详细分析 - commented out for production use
                # Uncomment the following block for debugging
                # if dataset_info['name'] == 'SEED7' and debug_sample_count < max_debug_samples and master_process:
                #     # 获取原始token序列用于分析
                #     raw_tokens = t.tolist()
                #     
                #     # 分析token级别信息（tiktoken已在文件顶部导入）
                #     enc = tiktoken.get_encoding("gpt2")
                #     
                #     words = pred_string.split(' ')
                #     print(f"\n{'='*80}")
                #     print(f"DEBUG Sample #{debug_sample_count + 1} - SEED7 Comprehensive Analysis")
                #     print(f"{'='*80}")
                #     print(f"Generated text: {pred_string}")
                #     print(f"Text length: {len(pred_string)} characters")
                #     print(f"Total words: {len(words)}")
                #     print(f"\n--- Word-level Analysis ---")
                #     for idx, word in enumerate(words):
                #         marker = " ← ANSWER" if word.startswith('(') and word.endswith(')') and word in dataset_info['label_dic'] else ""
                #         print(f"  [{idx:2d}] '{word}'{marker}")
                #     
                #     print(f"\n--- Token-level Analysis ---")
                #     print(f"Total tokens: {len(raw_tokens)}")
                #     # 找到Answer:之后的部分
                #     answer_start_idx = None
                #     for idx, word in enumerate(words):
                #         if word == 'Answer:':
                #             answer_start_idx = idx
                #             break
                #     
                #     if answer_start_idx is not None:
                #         print(f"Answer: found at word index {answer_start_idx}")
                #         print(f"Words after 'Answer:': {words[answer_start_idx+1:]}")
                #         
                #         # 尝试找到答案标签的位置
                #         answer_found = False
                #         for idx in range(answer_start_idx + 1, len(words)):
                #             word = words[idx]
                #             if word.startswith('(') and word.endswith(')') and word in dataset_info['label_dic']:
                #                 print(f"✓ Answer label '{word}' found at word index {idx}")
                #                 if dataset_info['result_idx'] is None:
                #                     dataset_info['result_idx'] = idx
                #                     print(f"  → Setting result_idx to {idx} (first successful detection)")
                #                 answer_found = True
                #                 break
                #         
                #         if not answer_found:
                #             print(f"⚠️  WARNING: No valid answer label found after 'Answer:'!")
                #             print(f"  This indicates a generation problem, not an index problem.")
                #             print(f"  Generated answer part: {' '.join(words[answer_start_idx+1:])}")
                #             
                #             # 分析为什么没有生成完整的答案
                #             answer_part_text = ' '.join(words[answer_start_idx+1:])
                #             if '<|endoftext|>' in answer_part_text:
                #                 print(f"  → Model generated '<|endoftext|>' before completing the answer")
                #                 print(f"  → This suggests the model may need more training or the generation stopped early")
                #     
                #     # 分析token序列
                #     print(f"\n--- Token Sequence Analysis (last 10 tokens) ---")
                #     if len(raw_tokens) >= 10:
                #         last_tokens = raw_tokens[-10:]
                #         for idx, token_id in enumerate(last_tokens):
                #             try:
                #                 token_text = enc.decode([token_id])
                #                 print(f"  Token {len(raw_tokens)-10+idx:3d} (ID {token_id:5d}): '{repr(token_text)}'")
                #             except:
                #                 print(f"  Token {len(raw_tokens)-10+idx:3d} (ID {token_id:5d}): <decode_error>")
                #     
                #     # 检查是否有endoftext token
                #     endoftext_id = 50256  # GPT-2 endoftext token ID
                #     if endoftext_id in raw_tokens:
                #         endoftext_pos = raw_tokens.index(endoftext_id)
                #         print(f"\n--- Endoftext Token Analysis ---")
                #         print(f"<|endoftext|> token (ID {endoftext_id}) found at token position {endoftext_pos}")
                #         print(f"Total tokens before <|endoftext|>: {endoftext_pos}")
                #         if endoftext_pos < len(raw_tokens) - 1:
                #             print(f"⚠️  WARNING: Tokens exist after <|endoftext|> token!")
                #             print(f"  This may cause decoding issues. Tokens after: {raw_tokens[endoftext_pos+1:]}")
                #     
                #     # 使用debug模式获取预测结果
                #     pred, debug_info = get_pred(pred_string, dataset_info, debug=True)
                #     print(f"\n--- Prediction Analysis ---")
                #     print(f"Debug info: {debug_info}")
                #     print(f"Predicted label: {pred} ({dataset_info['label_names'][pred] if pred != -1 and pred < len(dataset_info['label_names']) else 'FAILED'})")
                #     print(f"Actual label: {label[i].item()} ({dataset_info['label_names'][label[i].item()]})")
                #     
                #     # 如果result_idx已设置，验证它
                #     if dataset_info['result_idx'] is not None:
                #         if len(words) > dataset_info['result_idx']:
                #             word_at_idx = words[dataset_info['result_idx']]
                #             print(f"\n--- Result Index Verification ---")
                #             print(f"Configured result_idx: {dataset_info['result_idx']}")
                #             print(f"Word at result_idx {dataset_info['result_idx']}: '{word_at_idx}'")
                #             
                #             # 检查是否能从该位置提取有效标签（鲁棒性检查）
                #             extracted = None
                #             if word_at_idx in dataset_info['label_dic']:
                #                 extracted = word_at_idx
                #             elif word_at_idx.startswith('(') and len(word_at_idx) >= 3:
                #                 candidate = word_at_idx[:3]
                #                 if candidate in dataset_info['label_dic']:
                #                     extracted = candidate
                #             else:
                #                 # 尝试在字符串中搜索（处理 "(C))" 这种情况）
                #                 for label_key in dataset_info['label_dic'].keys():
                #                     if label_key in word_at_idx:
                #                         extracted = label_key
                #                         break
                #             
                #             if extracted:
                #                 if extracted == word_at_idx:
                #                     print(f"✓ result_idx {dataset_info['result_idx']} is CORRECT (exact match: '{extracted}')")
                #                 else:
                #                     print(f"⚠ result_idx {dataset_info['result_idx']} can extract label '{extracted}' from '{word_at_idx}' (robust extraction works)")
                #             else:
                #                 print(f"✗ result_idx {dataset_info['result_idx']} is INCORRECT (word is '{word_at_idx}', no valid label found)")
                #     
                #     # 如果实际找到的位置与result_idx不同，打印警告
                #     if debug_info.get('label_position') is not None:
                #         if dataset_info['result_idx'] is None:
                #             dataset_info['result_idx'] = debug_info['label_position']
                #             print(f"\n→ Auto-setting result_idx to {debug_info['label_position']} based on debug_info")
                #         elif debug_info['label_position'] != dataset_info['result_idx']:
                #             print(f"\n⚠️  WARNING: Actual answer position ({debug_info['label_position']}) differs from configured result_idx ({dataset_info['result_idx']})!")
                #             print(f"   Consider updating result_idx to {debug_info['label_position']}")
                #     
                #     print(f"{'='*80}\n")
                #     debug_sample_count += 1
                
                # 简单的调试输出：验证 SLEEP 数据集的 result_idx
                # Commented out for production use - uncomment for debugging
                # if dataset_info['name'] == 'SLEEP' and len(preds_raw) < 3 and master_process:
                #     words = pred_string.split(' ')
                #     result_idx = dataset_info.get('result_idx', None)
                #     if result_idx is not None and len(words) > result_idx:
                #         # 使用 debug 模式获取提取信息
                #         _, debug_info = get_pred(pred_string, dataset_info, debug=True)
                #         
                #         print(f"\n{'='*60}")
                #         print(f"SLEEP Debug Sample #{len(preds_raw) + 1} - Result Index Verification")
                #         print(f"{'='*60}")
                #         print(f"Generated text: {pred_string[:150]}...")  # 只显示前150个字符
                #         print(f"\nToken positions around result_idx={result_idx}:")
                #         start_idx = max(0, result_idx - 3)
                #         end_idx = min(len(words), result_idx + 4)
                #         for idx in range(start_idx, end_idx):
                #             marker = " ← result_idx" if idx == result_idx else ""
                #             print(f"  [{idx:2d}] '{words[idx]}'{marker}")
                #         
                #         # 显示提取到的 label 信息
                #         print(f"\n--- Label Extraction Info ---")
                #         print(f"Extraction method: {debug_info.get('method', 'unknown')}")
                #         if debug_info.get('original_word'):
                #             print(f"Original word at result_idx: '{debug_info['original_word']}'")
                #         if debug_info.get('found_label'):
                #             print(f"Extracted label: '{debug_info['found_label']}'")
                #             # 检查是否有鲁棒性问题（原始单词与提取的标签不同）
                #             if debug_info.get('original_word') and debug_info['original_word'] != debug_info['found_label']:
                #                 print(f"⚠️  ROBUSTNESS ISSUE DETECTED!")
                #                 print(f"   Original word: '{debug_info['original_word']}'")
                #                 print(f"   Extracted label: '{debug_info['found_label']}'")
                #                 print(f"   (Model generated something like '(A))' instead of '(A)', but extraction worked)")
                #             elif debug_info.get('original_word') and debug_info['original_word'] == debug_info['found_label']:
                #                 print(f"✓ Clean extraction (no robustness issue)")
                #         else:
                #             print(f"⚠️  WARNING: No valid label extracted!")
                #         print(f"{'='*60}\n")
                
                # 获取预测结果
                pred = get_pred(pred_string, dataset_info, debug=False)
                
                # 保存原始预测标签用于混淆矩阵
                preds_raw.append(pred)
                
                # 保存epoch_id（如果使用投票机制）
                if use_voting:
                    epoch_id = batch_epoch_ids[i]
                    epoch_ids.append(epoch_id)
                
                if not dataset_info['is_binary']:
                    # 多分类：转换为 one-hot 编码用于指标计算
                    if pred != -1:
                        pred_onehot = np.eye(dataset_info['num_classes'])[pred]
                    else:
                        # 如果预测失败，使用全零向量
                        pred_onehot = np.zeros(dataset_info['num_classes'])
                    preds.append(pred_onehot)
                else:
                    # 二分类：保持原始标签
                    if pred == -1:
                        pred = 0  # 默认值
                    preds.append(pred)

            targets.append(label)
    
    model.train()

    targets = torch.cat(targets, dim=0).numpy()
    preds = np.array(preds)
    preds_raw = np.array(preds_raw)
    
    # 如果使用投票机制（SEED7），进行视频级别投票
    if use_voting and len(epoch_ids) > 0:
        # 进行投票，得到视频级别的预测
        video_results = vote_predictions(preds_raw.tolist(), targets.tolist(), epoch_ids)
        
        # 提取视频级别的预测和真实标签
        video_preds = [r['video_pred'] for r in video_results]
        video_targets = [r['video_target'] for r in video_results]
        vote_scores = [r['vote_score'] for r in video_results]
        
        # 过滤掉预测失败的视频
        valid_video_mask = np.array([p != -1 for p in video_preds])
        if valid_video_mask.sum() > 0:
            video_preds_valid = np.array([video_preds[i] for i in range(len(video_preds)) if valid_video_mask[i]])
            video_targets_valid = np.array([video_targets[i] for i in range(len(video_targets)) if valid_video_mask[i]])
            
            # 转换预测为one-hot（作为概率），目标保持整数格式
            # get_metrics 期望：preds 是 one-hot (n_samples, n_classes)，targets 是整数 (n_samples,)
            video_preds_onehot = np.array([np.eye(dataset_info['num_classes'])[p] for p in video_preds_valid])
            
            # 计算视频级别的指标（使用投票后的预测）
            results = get_metrics(video_preds_onehot, video_targets_valid, dataset_info['metrics'], dataset_info['is_binary'])
            
            # 计算视频级别的混淆矩阵
            all_labels = np.arange(dataset_info['num_classes'])
            video_cm = confusion_matrix(video_targets_valid, video_preds_valid, labels=all_labels)
            results['video_confusion_matrix'] = video_cm.tolist()
            results['video_confusion_matrix_labels'] = all_labels.tolist()
            
            # 计算subject级别的准确率
            subject_results = {}
            for video_result in video_results:
                subject_id = video_result['subject_id']
                if subject_id not in subject_results:
                    subject_results[subject_id] = {
                        'correct': 0.0,
                        'total': 0
                    }
                subject_results[subject_id]['correct'] += video_result['vote_score']
                subject_results[subject_id]['total'] += 1
            
            # 计算每个subject的准确率
            subject_accuracies = {}
            for subject_id, stats in subject_results.items():
                subject_accuracies[subject_id] = stats['correct'] / stats['total']
                results[f'subject_{subject_id}_accuracy'] = subject_accuracies[subject_id]
                results[f'subject_{subject_id}_total_videos'] = stats['total']
            
            # 计算平均subject准确率
            if len(subject_accuracies) > 0:
                results['mean_subject_accuracy'] = np.mean(list(subject_accuracies.values()))
            
            # 添加投票统计信息
            results['voting_stats'] = {
                'total_videos': len(video_results),
                'valid_videos': int(valid_video_mask.sum()),
                'num_subjects': len(subject_results),
                'mean_vote_score': float(np.mean(vote_scores)) if len(vote_scores) else 0.0,
                'perfect_votes': sum(1 for s in vote_scores if s == 1.0),
                'tie_votes': sum(1 for s in vote_scores if s == 0.5),
                'wrong_votes': sum(1 for s in vote_scores if s == 0.0)
            }
            
            # 仍然保留chunk级别的混淆矩阵（用于对比）
            valid_mask = preds_raw != -1
            num_valid = valid_mask.sum()
            num_total = len(preds_raw)
            if num_valid > 0:
                targets_valid = targets[valid_mask]
                preds_raw_valid = preds_raw[valid_mask]
                chunk_cm = confusion_matrix(targets_valid, preds_raw_valid, labels=all_labels)
                results['chunk_confusion_matrix'] = chunk_cm.tolist()
                results['chunk_confusion_matrix_labels'] = all_labels.tolist()
                results['chunk_pred_success_rate'] = num_valid / num_total
        else:
            results = {}
            results['video_confusion_matrix'] = None
            results['video_confusion_matrix_labels'] = None
            print(f"Warning: All video predictions failed for {dataset_info['name']}. Total videos: {len(video_results)}")
    else:
        # 不使用投票机制，使用原有的chunk级别评估
        results = get_metrics(preds, targets, dataset_info['metrics'], dataset_info['is_binary'])
        
        # 对于多分类任务，计算混淆矩阵
        if not dataset_info['is_binary']:
            # 过滤掉预测失败的样本（pred == -1）
            valid_mask = preds_raw != -1
            num_valid = valid_mask.sum()
            num_total = len(preds_raw)
            
            if num_valid > 0:
                targets_valid = targets[valid_mask]
                preds_raw_valid = preds_raw[valid_mask]
                
                # 计算混淆矩阵
                # 获取所有类别标签（包括预测和真实标签）
                all_labels = np.arange(dataset_info['num_classes'])
                cm = confusion_matrix(targets_valid, preds_raw_valid, labels=all_labels)
                results['confusion_matrix'] = cm.tolist()
                results['confusion_matrix_labels'] = all_labels.tolist()
                results['pred_success_rate'] = num_valid / num_total  # 添加预测成功率
            else:
                results['confusion_matrix'] = None
                results['confusion_matrix_labels'] = None
                results['pred_success_rate'] = 0.0
                print(f"Warning: All predictions failed for {dataset_info['name']}. Total samples: {num_total}")

    return results


@torch.no_grad()
def evaluate_probs(model, dataset_info, dataloader):
    """LOSO 评估：对答案位置的 logits，在各类别 token 上做 softmax，得到每个样本的类别概率。

    注意：这里 model 应传入未包 DDP 的 raw_model（与原 evaluate 一致）。
    评估用的 val/test dataloader 使用 SequentialSampler，各 rank 看到完整数据且结果一致，
    因此每个 rank 计算得到相同结果，最终只由 master_process 落盘。

    返回 payload dict:
      sample_id (N,) str, subject_id (N,) int, y_true (N,) int,
      y_pred (N,) int(=argmax(y_prob)), y_prob (N, C) float
    """
    model.eval()
    class_token_ids = dataset_info['class_token_ids']
    answer_pos = dataset_info['answer_pos']             # k：答案 token 在 prompt/answer 里的位置
    cti = torch.as_tensor(class_token_ids, device=device, dtype=torch.long)

    all_sid, all_subj, all_true, all_prob = [], [], [], []
    for batch in dataloader:
        # MotorLoader(is_val=True) 返回 9 元组，末两个是 sample_id / subject_id
        X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask, sample_id, subject_id = batch

        X_eeg = X_eeg.float().to(device, non_blocking=True)
        X_text = X_text.to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)
        if input_mask is not None:
            input_mask = input_mask.to(device, non_blocking=True)

        # 只喂到答案位置之前的“共同前缀”（X_text 在验证集里就是完整 prompt，长度 p）。
        # answer_pos == p 时为无操作；== p-1 时会去掉尾部 '(' token。相应裁剪 eeg-text 注意力掩码。
        E = X_eeg.size(1)                                # eeg token 数（与掩码 eeg 段一致）
        X_text = X_text[:, :answer_pos]
        gpt_mask = gpt_mask[:, :, :E + answer_pos, :E + answer_pos]

        with ctx:
            # 复用 NeuroLM.forward：y_eeg=y_text=None -> 只返回最后一个位置的 logits，
            # 即“下一个 token”（答案字母）的分布。等价于 generate 的第 0 步。
            _, _, logits = model(
                x_eeg=X_eeg, y_eeg=None, x_text=X_text, y_text=None,
                input_chans=input_chans, input_time=input_time,
                input_mask=input_mask, eeg_mask=None, eeg_text_mask=gpt_mask)

        last_logits = logits[:, -1, :].float()          # [B, vocab]
        class_logits = last_logits[:, cti]              # [B, C]
        prob = torch.softmax(class_logits, dim=-1)      # [B, C]

        all_prob.append(prob.cpu().numpy())
        if torch.is_tensor(label):
            all_true.append(label.cpu().numpy().reshape(-1))
        else:
            all_true.append(np.asarray(label).reshape(-1))
        all_sid.extend([str(s) for s in sample_id])
        if torch.is_tensor(subject_id):
            all_subj.extend([int(s) for s in subject_id.cpu().numpy().reshape(-1)])
        else:
            all_subj.extend([int(s) for s in subject_id])

    model.train()

    y_prob = np.concatenate(all_prob, axis=0).astype(np.float32)
    y_true = np.concatenate(all_true, axis=0).astype(np.int64)
    y_pred = y_prob.argmax(axis=1).astype(np.int64)
    sample_id = np.asarray(all_sid, dtype=object)
    subject_id = np.asarray(all_subj, dtype=np.int64)

    assert len(sample_id) == len(y_true) == len(y_pred) == y_prob.shape[0] == len(subject_id), \
        "evaluate_probs: 各数组长度不一致"
    return {
        'sample_id': sample_id,
        'subject_id': subject_id,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }


def save_fold_results(args, name, payload, fold_stats):
    """按 sample_id 排序后写出 {task}_{model}_fold{i:02d}.npz 与 .json。

    失败直接抛错退出（不静默跳过）。仅在 master_process 调用。
    """
    task = args.task_name if args.task_name else name.lower()
    model_name = args.model_name
    results_dir = args.results_dir if args.results_dir else args.out_dir
    os.makedirs(results_dir, exist_ok=True)
    base = f"{task}_{model_name}_fold{args.fold:02d}"
    npz_path = os.path.join(results_dir, base + '.npz')
    json_path = os.path.join(results_dir, base + '.json')

    # 按 sample_id 稳定排序，保证可复现
    order = np.argsort(payload['sample_id'].astype('U'), kind='stable')
    sample_id = payload['sample_id'][order].astype('U')
    y_true = payload['y_true'][order].astype(np.int64)
    y_pred = payload['y_pred'][order].astype(np.int64)
    y_prob = payload['y_prob'][order].astype(np.float32)
    subject_id = payload['subject_id'][order].astype(np.int64)

    fold_bacc = float(balanced_accuracy_score(y_true, y_pred))

    np.savez(
        npz_path,
        sample_id=sample_id,      # (N,) str
        y_true=y_true,            # (N,) int
        y_pred=y_pred,            # (N,) int (argmax)
        y_prob=y_prob,            # (N, C) float, softmax
        subject_id=subject_id,    # (N,) int
    )

    meta = {
        'fold': int(args.fold),
        'test_subject': fold_stats['test_subject'],
        'val_subject': fold_stats['val_subject'],
        'balanced_accuracy': fold_bacc,
        'best_epoch': int(fold_stats['best_epoch']),
        'best_val_balanced_accuracy': float(fold_stats['best_val_bacc']),
        'num_samples': int(len(sample_id)),
        'num_classes': int(y_prob.shape[1]),
        'task': task,
        'model': model_name,
        'hyperparams': fold_stats['hyperparams'],
        'train_time_sec': float(fold_stats['train_time_sec']),
        'peak_gpu_mem_mb': float(fold_stats['peak_gpu_mem_mb']),
        'gpu_name': fold_stats['gpu_name'],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(meta), f, ensure_ascii=False, indent=2)

    # 落盘自检
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        raise RuntimeError(f"结果文件保存失败: {npz_path} / {json_path}")
    print(f"[fold {args.fold}] saved: {npz_path}")
    print(f"[fold {args.fold}] saved: {json_path}")
    print(f"[fold {args.fold}] test balanced_accuracy = {fold_bacc:.4f} (best_epoch={fold_stats['best_epoch']})")


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--text_data_dir', default='./text', help='path where text data (train.bin, val.bin) is')
    parser.add_argument('--dataset_dir', default='./', help='path of eeg dataset')
    parser.add_argument('--chan_size', default=30, type=int, help='channel number. this is parameter for Custom Stress Dataset')
    parser.add_argument('--tokenizer_path', default='checkpoints/VQ.py', help='path where tokenizer is')
    parser.add_argument('--NeuroLM_path', default='checkpoints/NeuroLM-B.pt', help='path where NeuroLM model is')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='NeuroLM')
    parser.add_argument('--wandb_runname', default='instruction-B')
    parser.add_argument('--wandb_api_key', type=str)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=64, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='weight decay (default: 1e-1)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    # leave-one-subject-out / 结果保存相关（仅在 --fold 指定时启用 LOSO 逻辑）
    parser.add_argument('--fold', default=None, type=int,
                        help='LOSO 折号（0..N-1）。指定后启用按被试留一的分折与结果保存；不指定则维持原行为')
    parser.add_argument('--n_folds', default=None, type=int,
                        help='折数上限，默认 None 即等于被试数 N')
    parser.add_argument('--model_name', default='NeuroLM', type=str,
                        help='用于结果文件名 {task}_{model}_fold{i}.npz/json 中的模型标识')
    parser.add_argument('--task_name', default=None, type=str,
                        help='结果文件名中的任务标识，默认取数据集名小写（如 motor）')
    parser.add_argument('--results_dir', default=None, type=str,
                        help='npz/json 结果保存目录，默认 out_dir')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)