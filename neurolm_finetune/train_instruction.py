"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import os
import time
import json
import yaml
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sklearn.metrics import confusion_matrix

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from pathlib import Path
import tiktoken
from utils import prepare_KaggleERN_dataset, prepare_STRESS_dataset, prepare_TUAB_dataset, prepare_TUEV_dataset, prepare_TUSL_dataset, prepare_HMC_dataset, prepare_Workload_dataset, prepare_SEED7_dataset, prepare_motor_dataset, cosine_scheduler, get_metrics #, prepare_KaggleERN_dataset
from downstream_dataset import SEEDDataset
from torch.utils.data.dataset import ConcatDataset


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


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


def get_instruct_datasets(args, downstream_dataset: str, eeg_max_len=-1, text_max_len=-1):
        dataset_info = {'name': downstream_dataset}
        if downstream_dataset == 'SEED':
            dataset_train = SEEDDataset(Path(args.dataset_dir, 'h5data/seed-3.hdf5'), window_size=800, stride_size=800, trial_start_percentage=0, 
                                        trial_end_percentage=0.6, is_instruct=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            dataset_val = SEEDDataset(Path(args.dataset_dir, 'h5data/seed-3.hdf5'), window_size=800, stride_size=800, trial_start_percentage=0.6, 
                                    trial_end_percentage=0.8, is_instruct=True, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            dataset_test = SEEDDataset(Path(args.dataset_dir, 'h5data/seed-3.hdf5'), window_size=800, stride_size=800, trial_start_percentage=0.8, 
                                    trial_end_percentage=1, is_instruct=True, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            
            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 3
            dataset_info['result_idx'] = 11
            dataset_info['label_dic'] = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        elif downstream_dataset == 'KaggleERN':
            dataset_train, dataset_test, dataset_val = prepare_KaggleERN_dataset(Path(args.dataset_dir), chan_size=args.chan_size, is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            #TODO: 待修改
            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 9
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}
        elif downstream_dataset == 'KaggleERN':
            dataset_train, dataset_test, dataset_val = prepare_KaggleERN_dataset(Path(args.dataset_dir), chan_size=args.chan_size, is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

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
        elif downstream_dataset == 'TUAB':
            dataset_train, dataset_test, dataset_val = prepare_TUAB_dataset(Path(args.dataset_dir, 'TUAB/processed'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 7
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}
        elif downstream_dataset == 'TUEV':
            dataset_train, dataset_test, dataset_val = prepare_TUEV_dataset(Path(args.dataset_dir, 'TUEV'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 6
            dataset_info['result_idx'] = 34
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5}
        elif downstream_dataset == 'TUSL':
            dataset_train, dataset_test, dataset_val = prepare_TUSL_dataset(Path(args.dataset_dir, 'TUSL'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 3
            dataset_info['result_idx'] = 17
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2}
        elif downstream_dataset == 'HMC':
            dataset_train, dataset_test, dataset_val = prepare_HMC_dataset(Path(args.dataset_dir, 'HMC'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 5
            dataset_info['result_idx'] = 22
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4}
        elif downstream_dataset == 'Workload':
            dataset_train, dataset_test, dataset_val = prepare_Workload_dataset(Path(args.dataset_dir, 'EEGWorkload'), is_instruct=True, 
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
            dataset_info['num_classes'] = 7
            dataset_info['result_idx'] = 24  # Answer: (A) or (B) etc. position in the generated text (0-indexed)
            # Generated text format: "Question: ... Options: (A) ... (G) ... Answer: (X) <|endoftext|>)"
            # Answer "(X)" is at word index 24 (after splitting by space)
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5, '(G)': 6}
            dataset_info['label_names'] = ['happy', 'sad', 'neutral', 'disgust', 'fear', 'surprise', 'anger']
        elif downstream_dataset == 'MOTOR':
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

        dataset_info['dataset_train'] = dataset_train
        dataset_info['dataset_val'] = dataset_val
        dataset_info['dataset_test'] = dataset_test

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
    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset_dir}\n"
            f"Path must contain: ['stress', 'kaggleern', 'seed', 'motor']"
        )
    all_datasets.append(get_instruct_datasets(args, name, eeg_max_len=276, text_max_len=80))
        
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
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=int(args.warmup_ratio * num_training_steps_per_epoch * args.epochs)
    )

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
    for epoch in range(start_epoch, args.epochs):
        for dataset_info in datasets:
            if args.eval_only:
                break
            for step, (batch) in enumerate(dataset_info['data_loader_train']):
                # determine and set the learning rate for this iteration
                lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
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
        
        if master_process and (not args.eval_only):
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
        
        # validation and test
        for dataset_info in all_datasets:
            print('Dataset:', dataset_info['name'])
            results_val = evaluate(raw_model, dataset_info, dataset_info['data_loader_val'], decode)
            print('=' * 10)
            print('Eval:')
            for metric in results_val.keys():
                if metric not in ['confusion_matrix', 'confusion_matrix_labels']:
                    print(metric + ':', results_val[metric])
            # 打印混淆矩阵（如果是多分类任务）
            if results_val.get('confusion_matrix') is not None:
                print('Confusion Matrix (Val):')
                cm_val = np.array(results_val['confusion_matrix'])
                labels_val = results_val.get('confusion_matrix_labels', list(range(len(cm_val))))
                # 获取类别名称（如果有的话）
                if 'label_names' in dataset_info:
                    label_names = dataset_info['label_names']
                else:
                    label_names = [f'Class {i}' for i in labels_val]
                print(f'Labels: {label_names}')
                print(cm_val)
            results_test = evaluate(raw_model, dataset_info, dataset_info['data_loader_test'], decode)
            print('=' * 10)
            print('Test:')
            for metric in results_test.keys():
                if metric not in ['confusion_matrix', 'confusion_matrix_labels']:
                    print(metric + ':', results_test[metric])
            # 打印混淆矩阵（如果是多分类任务）
            if results_test.get('confusion_matrix') is not None:
                print('Confusion Matrix (Test):')
                cm_test = np.array(results_test['confusion_matrix'])
                labels_test = results_test.get('confusion_matrix_labels', list(range(len(cm_test))))
                # 获取类别名称（如果有的话）
                if 'label_names' in dataset_info:
                    label_names = dataset_info['label_names']
                else:
                    label_names = [f'Class {i}' for i in labels_test]
                print(f'Labels: {label_names}')
                print(cm_test)
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
                    f.write(json.dumps(local_log_data) + '\n')
                    
            if args.wandb_log and master_process:
                wandb_log_data = {}
                for metric in results_val.keys():
                    # 跳过混淆矩阵和标签（它们已经在日志文件中记录，wandb不支持直接记录二维数组）
                    if metric not in ['confusion_matrix', 'confusion_matrix_labels']:
                        wandb_log_data['val_' + dataset_info['name'] + '/' + metric] = results_val[metric]
                        wandb_log_data[f'test_' + dataset_info['name'] + '/' + metric] = results_test[metric]
                wandb.log(wandb_log_data)
                
                # 如果有混淆矩阵，可以记录预测成功率
                if results_val.get('pred_success_rate') is not None:
                    wandb.log({
                        'val_' + dataset_info['name'] + '/pred_success_rate': results_val['pred_success_rate'],
                        'test_' + dataset_info['name'] + '/pred_success_rate': results_test.get('pred_success_rate', 0.0)
                    })
        if args.eval_only:
            break

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
            # 首先尝试按 result_idx 位置解析
            words = pred_string.split(' ')
            if len(words) > dataset_info['result_idx']:
                pred_word = words[dataset_info['result_idx']]
                debug_info['method'] = 'result_idx'
                debug_info['label_position'] = dataset_info['result_idx']
                debug_info['found_label'] = pred_word[:3] if pred_word.startswith('(') else pred_word
                
                if pred_word.startswith('('):
                    pred_word = pred_word[:3]
                pred = dataset_info['label_dic'][pred_word]
                debug_info['found_label'] = pred_word
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

@torch.no_grad()
def evaluate(model, dataset_info, dataloader, decode):
    model.eval()
    preds = []
    preds_raw = []  # 保存原始预测标签用于混淆矩阵
    targets = []
    for _, (batch) in enumerate(dataloader):
        X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
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
                pred_string = decode(t.tolist())
                
                # 获取预测结果
                pred = get_pred(pred_string, dataset_info, debug=False)
                
                # 保存原始预测标签用于混淆矩阵
                preds_raw.append(pred)
                
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

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
