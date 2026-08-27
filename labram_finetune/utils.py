# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import re
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf
import h5py

from tensorboardX import SummaryWriter
# from data_processor.dataset import ShockDataset
import pickle
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data.dataloader import default_collate


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
            
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)
    
    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)
            
    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict): 
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
        
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")
        
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    
    # Debug: 打印关键参数
    total_iters = epochs * niter_per_ep
    print(f"[DEBUG cosine_scheduler] epochs={epochs}, niter_per_ep={niter_per_ep}, total_iters={total_iters}, warmup_iters={warmup_iters}")
    
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        print(f"[DEBUG cosine_scheduler] warmup_schedule length: {len(warmup_schedule)}")

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    print(f"[DEBUG cosine_scheduler] iters length: {len(iters)}, range: epochs*niter_per_ep - warmup_iters = {total_iters} - {warmup_iters} = {total_iters - warmup_iters}")
    
    if len(iters) == 0:
        print(f"[WARNING cosine_scheduler] iters is empty! This means epochs*niter_per_ep ({total_iters}) <= warmup_iters ({warmup_iters})")
        schedule = np.array([])
    else:
        schedule = np.array(
            [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
        print(f"[DEBUG cosine_scheduler] schedule length: {len(schedule)}")

    schedule = np.concatenate((warmup_schedule, schedule))
    print(f"[DEBUG cosine_scheduler] final schedule length: {len(schedule)}, expected: {total_iters}")

    if len(schedule) != epochs * niter_per_ep:
        print(f"[ERROR cosine_scheduler] Schedule length mismatch! Got {len(schedule)}, expected {epochs * niter_per_ep}")
        print(f"[ERROR cosine_scheduler] This will cause issues. Please check your dataset size and batch size.")
    else:
        assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None, save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)
                
            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
        
        # 删除所有非 best 的 checkpoint 文件，只保留 best
        # 这样可以节省存储空间，只保留最佳模型
        if is_main_process():
            all_checkpoints = glob.glob(str(output_dir / 'checkpoint*.pth'))
            for ckpt_path in all_checkpoints:
                ckpt_file = Path(ckpt_path)
                # 只保留包含 'best' 的 checkpoint，删除其他所有 checkpoint（包括 checkpoint.pth 和 checkpoint-{数字}.pth）
                if 'best' not in ckpt_file.name.lower():
                    try:
                        ckpt_file.unlink()
                        print(f"已删除旧 checkpoint: {ckpt_file.name}")
                    except Exception as e:
                        print(f"删除 checkpoint {ckpt_file.name} 时出错: {e}")
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)           

def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)
    
    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            
            # 处理模型结构不匹配的问题（特别是head部分）
            checkpoint_model = checkpoint['model']
            model_state_dict = model_without_ddp.state_dict()
            
            # 检查是否有键名不匹配的问题
            checkpoint_keys = set(checkpoint_model.keys())
            model_keys = set(model_state_dict.keys())
            
            # 如果checkpoint中有旧的head.fc.*格式，而模型使用新的head.fc_layer*格式
            old_head_keys = [k for k in checkpoint_keys if k.startswith('head.fc.')]
            new_head_keys = [k for k in model_keys if k.startswith('head.fc_layer')]
            
            # 如果head结构完全不匹配，跳过resume，让finetune参数生效
            if old_head_keys and new_head_keys:
                print(f"Warning: Checkpoint uses old head structure (head.fc.*), model uses new structure (head.fc_layer*).")
                print(f"Skipping resume to use finetune checkpoint instead.")
                print(f"Note: This means training will start from epoch 0, not resuming from previous checkpoint.")
                args.resume = ''  # 清空resume，让finetune参数生效
                return  # 跳过resume，finetune已经在之前加载过了
            
            # 尝试加载，如果还有不匹配的键，使用strict=False
            try:
                model_without_ddp.load_state_dict(checkpoint_model, strict=True)
            except RuntimeError as e:
                if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                    print(f"Warning: Some keys don't match, loading with strict=False")
                    print(f"Error details: {e}")
                    # 只加载匹配的键
                    model_without_ddp.load_state_dict(checkpoint_model, strict=False)
                else:
                    raise
            
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1#checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


# def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
#     shock_dataset_list = []
#     ch_names_list = []
#     for dataset_list, window_size in zip(datasets, time_window):
#         dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
#         shock_dataset_list.append(dataset)
#         ch_names_list.append(dataset.get_ch_names())
#     return shock_dataset_list, ch_names_list


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, data_key='X', label_key='y'):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.data_key = data_key
        self.label_key = label_key

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root, self.files[index])
        sample = pickle.load(open(sample_path, "rb"))

        if self.data_key not in sample:
            raise KeyError(f"Data key '{self.data_key}' not found in {sample_path}")
        if self.label_key not in sample:
            raise KeyError(f"Label key '{self.label_key}' not found in {sample_path}")

        X = sample[self.data_key]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample[self.label_key]
        X = torch.FloatTensor(X)
        return X, Y
    
# ===== [LOSO] Stress 任务的 sample_id / subject_id 提取，与
# cbramod_finetune/datasets/custom_stress_dataset.py 的 compute_stress_sample_id /
# extract_subject_id_from_name 使用同一套正则和输出格式，保证同一批底层 pickle 文件
# （preprocessing/stress_preprocess.ipynb 生成的 Sub04_increase_edf27_chunk0012.pickle）
# 在各模型间算出来的 sample_id 完全一致。 =====
_STRESS_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(increase|normal)_edf(\d+)_chunk(\d+)$')
_STRESS_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def compute_stress_sample_id(chunk_id):
    """'Sub04_increase_edf27_chunk0012' -> 'S04_edf27_chunk0012'。"""
    m = _STRESS_SAMPLE_ID_RE.match(chunk_id)
    if not m:
        raise ValueError(f"Cannot parse chunk_id for sample_id: {chunk_id!r}")
    subject_num = int(m.group(1))
    edf_num = int(m.group(3))
    local_idx = int(m.group(4))
    return f"S{subject_num:02d}_edf{edf_num}_chunk{local_idx:04d}"


def extract_stress_subject_id(name):
    """'Sub04_increase_edf27_chunk0012.pickle'（或包含它的任意路径）-> 'Sub04'。"""
    m = _STRESS_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def list_stress_files_by_subject(root):
    """[LOSO] 扫描 root/{train,val,test}/*.pickle，按受试者分组（'Sub04' -> [path, ...]），
    用于构造受试者独立（LOSO）划分。"""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_stress_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.join(split_dir, fname))
    return subject_to_files


class StressLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, data_key='X', label_key='y', expected_channels=30,
                 return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.data_key = data_key
        self.label_key = label_key
        self.expected_channels = expected_channels
        self.return_sample_id = return_sample_id

        # [LOSO] sample_id 在数据集构造（列文件）时一次性确定，与 self.files 一一对应；
        # 不受 shuffle / batch_size / num_workers 影响。解析失败直接报错退出（不静默跳过）。
        if self.return_sample_id:
            self.sample_ids = [
                compute_stress_sample_id(os.path.splitext(os.path.basename(f))[0]) for f in self.files
            ]
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError(
                    "Duplicate sample_id detected in StressLoader; check for duplicate/conflicting chunk files."
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root, self.files[index])
        sample = pickle.load(open(sample_path, "rb"))

        X = sample[self.data_key]
        Y = sample[self.label_key]

        if X.shape[0] != self.expected_channels:
            return None

        if isinstance(Y, (list, np.ndarray)) and len(Y) > 0:
            Y = int(Y[0])
        elif isinstance(Y, (list, np.ndarray)) and len(Y) == 0:
            raise ValueError(f"Empty label in {sample_path}")
        else:
            Y = int(Y)

        # increase=1 / normal=0
        if Y not in (0, 1):
            raise ValueError(f"Invalid label {Y} in {sample_path}, expected 0 or 1")

        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)

        if self.return_sample_id:
            return torch.FloatTensor(X), Y, self.sample_ids[index]
        return torch.FloatTensor(X), Y


# [KaggleERN channel fix] 完整的 56 通道原始蒙太奇顺序，跟
# eegpt_finetune/linear_probe_EEGPT_KaggleERN.py 的 use_channels_names 一致
# （也是 preprocessing/preprocess_KaggleERN_new.ipynb 里 df_to_raw_full() 存盘时
# 的 CSV 列顺序）。labram/neurolm 现有的 55 通道 ch_names 列表比这份少一个 'PO8'——
# 核对过 labram 自己的 standard_1020 字典，PO8 明明在里面（Seed 任务的 ch_names
# 也在用它），说明当初漏抄了这一个，不是故意排除。但线上已经跑过并且产出过真实
# 结果的 s42_n55-labram/s42_n55-neurolm 数据本身就是 55 通道（历史 HPO 跑得通就是
# 证据——56 通道喂给只有 55 个位置编码的模型会在 pos_embed 那里直接 shape 对不上），
# 所以这里不改动模型实际吃的通道数，只是让 loader 在拿到 56 通道的原始数据（新
# smoke 数据就是这样)时，自动按名字丢掉 PO8 对齐到 55；已经是 55 通道的真实数据
# 完全不受影响。
KAGGLEERN_FULL_56_CH_NAMES = [
    'FP1', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2',
    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
    'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POZ', 'PO8', 'O1', 'O2',
]


def kaggleern_select_channel_idx(used_ch_names):
    """给定模型实际使用的通道名列表（如 labram 的 55 通道 ch_names），
    返回它们在 KAGGLEERN_FULL_56_CH_NAMES 里的下标列表，供 KaggleERNLoader
    在遇到 56 通道原始数据时按名字选出对应的 55 个通道。"""
    return [KAGGLEERN_FULL_56_CH_NAMES.index(name) for name in used_ch_names]


class KaggleERNLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, data_key='X', label_key='y',
                 return_sample_id=False, select_channel_idx=None):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.data_key = data_key
        self.label_key = label_key
        # [KaggleERN 通道修复] 只在原始数据是完整 56 通道时生效（见上面
        # kaggleern_select_channel_idx 的说明）；真实的 55 通道数据不受影响，
        # 不传这个参数（None）也完全不受影响，向后兼容。
        self.select_channel_idx = select_channel_idx
        # [KaggleERN bestval] 按最佳 epoch 权重做事后干净推理时用；不影响现有训练/
        # 评估行为（默认 False）。sample_id 就是文件名本身（去掉 .pickle），
        # preprocess_KaggleERN_new.ipynb 存盘时已经用 epoch_id（形如
        # "S02_Sess01_FB004"）当文件名，不需要像 Motor 那样再算一遍。
        self.return_sample_id = return_sample_id
        if self.return_sample_id:
            self.sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError(
                    "Duplicate sample_id detected in KaggleERNLoader; check for duplicate/conflicting epoch files."
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root, self.files[index])
        sample = pickle.load(open(sample_path, "rb"))

        if self.data_key not in sample.keys():
            raise KeyError(f"Data key '{self.data_key}' not found in {sample_path}")
        if self.label_key not in sample.keys():
            raise KeyError(f"Label key '{self.label_key}' not found in {sample_path}")

        X = sample[self.data_key]
        # [KaggleERN 通道修复] 只有真的是 56 通道原始数据时才做选择；已经是
        # 目标通道数（55）的真实数据原样通过，不受影响。
        if self.select_channel_idx is not None and X.shape[0] == len(KAGGLEERN_FULL_56_CH_NAMES):
            X = X[self.select_channel_idx]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample[self.label_key]
        X = torch.FloatTensor(X)
        if self.return_sample_id:
            return X, Y, self.sample_ids[index]
        return X, Y

# ===== [LOSO] 跨模型稳定的 sample_id，与 cbramod_finetune/datasets/motortask_dataset.py 的
# compute_sample_id、biot_finetune/utils.py 的 compute_motion_sample_id、
# eegpt_finetune/linear_probe_EEGPT_Motor.py 的 compute_sample_id 保持同一套算法。
# epoch_id 形如 "Sub04_Walkslow_epoch009"；纯函数，只依赖字符串本身，与 shuffle /
# batch_size / num_workers 无关。几个模型跑同一批底层 AllSubjects_Epochs pickle
# 文件时，算出来的 sample_id 必须完全一致。 =====
_MOTOR_SAMPLE_ID_TASK_ORDER = ['Walk', '8', 'Horizontal', 'Vertical', 'Pick', 'Stair']
_MOTOR_SAMPLE_ID_SPEED_ORDER = ['slow', 'medium', 'fast']
_MOTOR_SAMPLE_ID_TASK_OFFSET = 3000
_MOTOR_SAMPLE_ID_SPEED_OFFSET = 1000
_MOTOR_SAMPLE_ID_RE = re.compile(r'^Sub(\d+)_(.+?)_epoch(\d+)$')
_MOTOR_SUBJECT_RE = re.compile(r'(Sub\d+)_')


def _parse_motor_task_token(task_token):
    for speed in _MOTOR_SAMPLE_ID_SPEED_ORDER:
        if task_token.endswith(speed):
            return task_token[: -len(speed)], speed
    raise ValueError(f"Cannot parse speed suffix (slow/medium/fast) from task token: {task_token!r}")


def compute_motor_sample_id(epoch_id):
    """由 epoch_id（如 'Sub04_Walkslow_epoch009'）确定性地生成 sample_id，
    格式：S{subject:02d}_ep{index:05d}（Motor 无 session 概念，不加 sess 段）。"""
    m = _MOTOR_SAMPLE_ID_RE.match(epoch_id)
    if not m:
        raise ValueError(f"Cannot parse epoch_id for sample_id: {epoch_id!r}")
    subject_num = int(m.group(1))
    task_token = m.group(2)
    local_idx = int(m.group(3))
    base_task, speed = _parse_motor_task_token(task_token)
    if base_task not in _MOTOR_SAMPLE_ID_TASK_ORDER:
        raise ValueError(
            f"Unknown base task {base_task!r} parsed from epoch_id {epoch_id!r}; "
            f"expected one of {_MOTOR_SAMPLE_ID_TASK_ORDER}"
        )
    task_idx = _MOTOR_SAMPLE_ID_TASK_ORDER.index(base_task)
    speed_idx = _MOTOR_SAMPLE_ID_SPEED_ORDER.index(speed)
    global_index = task_idx * _MOTOR_SAMPLE_ID_TASK_OFFSET + speed_idx * _MOTOR_SAMPLE_ID_SPEED_OFFSET + local_idx
    if global_index > 99999:
        raise ValueError(f"sample_id index overflow (>99999) for epoch_id {epoch_id!r}: {global_index}")
    return f"S{subject_num:02d}_ep{global_index:05d}"


def extract_motor_subject_id(name):
    """'Sub04_8fast_epoch001.pickle'（或包含它的任意路径）-> 'Sub04'。"""
    m = _MOTOR_SUBJECT_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def list_motor_files_by_subject(root):
    """[LOSO] 扫描 root/{train,val,test}/*.pickle，按受试者分组（'Sub04' -> [path, ...]），
    用于构造受试者独立（LOSO）划分。"""
    subject_to_files = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".pickle"):
                continue
            sid = extract_motor_subject_id(fname)
            if sid is None:
                continue
            subject_to_files[sid].append(os.path.join(split_dir, fname))
    return subject_to_files


class MotorLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, data_key='X', label_key='y', expected_channels=20,
                 return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.data_key = data_key
        self.label_key = label_key
        self.expected_channels = expected_channels
        self.return_sample_id = return_sample_id

        # [LOSO] sample_id 在数据集构造（列文件）时一次性确定，与 self.files 一一对应；
        # 不受 shuffle / batch_size / num_workers 影响。解析失败直接报错退出（不静默跳过）。
        if self.return_sample_id:
            self.sample_ids = [
                compute_motor_sample_id(os.path.splitext(os.path.basename(f))[0]) for f in self.files
            ]
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError(
                    "Duplicate sample_id detected in MotorLoader; check for duplicate/conflicting epoch files."
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root, self.files[index])
        sample = pickle.load(open(sample_path, "rb"))

        X = sample[self.data_key]
        Y = sample[self.label_key]

        if X.shape[0] != self.expected_channels:
            return None

        # 标签已经是从0开始的索引 (0-5)，不需要再减1
        if isinstance(Y, (list, np.ndarray)) and len(Y) > 0:
            Y = int(Y[0])
        elif isinstance(Y, (list, np.ndarray)) and len(Y) == 0:
            raise ValueError(f"Empty label in {sample_path}")
        else:
            Y = int(Y)

        # 确保标签在 0-5 范围内（6 分类：0,1,2,3,4,5）
        if Y < 0 or Y > 5:
            raise ValueError(f"Invalid label {Y} in {sample_path}, expected 0-5")

        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)

        if self.return_sample_id:
            return torch.FloatTensor(X), Y, self.sample_ids[index]
        return torch.FloatTensor(X), Y

class SleepLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, data_key='signal', label_key='label', expected_channels=6,
                 return_sample_id=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.data_key = data_key
        self.label_key = label_key
        self.expected_channels = expected_channels
        self.return_sample_id = return_sample_id

        # sample_id 就是文件名本身（去掉 .pickle），preprocess_sleep.py 存盘时已经用
        # epoch_id（形如 "sub01_ep0000"）当文件名，不需要像 Motor 那样再算一遍。
        if self.return_sample_id:
            self.sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError(
                    "Duplicate sample_id detected in SleepLoader; check for duplicate/conflicting epoch files."
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root, self.files[index])
        sample = pickle.load(open(sample_path, "rb"))

        X = sample[self.data_key]
        Y = sample[self.label_key]

        if X.shape[0] != self.expected_channels:
            return None

        # Sleep数据集的标签已经是0-4（5分类），不需要转换
        if isinstance(Y, (list, np.ndarray)) and len(Y) > 0:
            Y = int(Y[0])
        elif isinstance(Y, (list, np.ndarray)) and len(Y) == 0:
            raise ValueError(f"Empty label in {sample_path}")
        else:
            Y = int(Y)

        # 确保标签在0-4范围内（5分类：0,1,2,3,4）
        if Y < 0 or Y > 4:
            raise ValueError(f"Invalid label {Y} in {sample_path}, expected 0-4")

        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)

        if self.return_sample_id:
            return torch.FloatTensor(X), Y, self.sample_ids[index]
        return torch.FloatTensor(X), Y

class SeedLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, data_key='signal', label_key='label', expected_channels=62):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.data_key = data_key
        self.label_key = label_key
        self.expected_channels = expected_channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample_path = os.path.join(self.root, self.files[index])
        sample = pickle.load(open(sample_path, "rb"))

        X = sample[self.data_key]
        Y = sample[self.label_key]

        if X.shape[0] != self.expected_channels:
            return None

        # Seed数据集的标签已经是0-6，不需要减1
        if isinstance(Y, (list, np.ndarray)) and len(Y) > 0:
            Y = int(Y[0])
        elif isinstance(Y, (list, np.ndarray)) and len(Y) == 0:
            raise ValueError(f"Empty label in {sample_path}")
        else:
            Y = int(Y)

        # 确保标签在0-6范围内
        if Y < 0 or Y > 6:
            raise ValueError(f"Invalid label {Y} in {sample_path}, expected 0-6")
        
        # 跳过neutral类别（label=2），并重新映射标签
        # 原始映射: 0=happy, 1=sad, 2=neutral, 3=disgust, 4=fear, 5=surprise, 6=anger
        # 新映射: 0=happy, 1=sad, 2=disgust, 3=fear, 4=surprise, 5=anger (跳过neutral)
        if Y == 2:  # neutral类别，跳过
            return None
        
        # 重新映射标签：0->0, 1->1, 3->2, 4->3, 5->4, 6->5
        label_mapping = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        Y = label_mapping[Y]

        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)

        # 获取epoch_id（用于视频级投票评估）
        epoch_id = sample.get('epoch_id', None)
        if epoch_id is None:
            # 如果pickle中没有epoch_id，从文件名提取
            filename = os.path.basename(self.files[index]).replace('.pickle', '')
            epoch_id = filename

        return torch.FloatTensor(X), Y, epoch_id

# 全局变量用于统计被跳过的样本数量（避免重复打印太多警告）
_skip_collate_stats = {'total_batches': 0, 'skipped_samples': 0, 'last_warning_step': -100}

def skip_failed_collate(batch):
    """
    Collate function that skips None samples.
    Note: This will cause variable batch sizes if samples are filtered.
    """
    global _skip_collate_stats
    original_size = len(batch)
    clean_batch = []
    skipped_count = 0
    
    for item in batch:
        if item is None:
            skipped_count += 1
            continue
        clean_batch.append(item)

    # 更新统计信息
    _skip_collate_stats['total_batches'] += 1
    _skip_collate_stats['skipped_samples'] += skipped_count

    # 如果全部样本都坏了，返回 None（但训练循环需要处理这种情况）
    if len(clean_batch) == 0:
        return None

    # 每 100 个 batch 打印一次统计信息，避免输出过多
    if _skip_collate_stats['total_batches'] - _skip_collate_stats['last_warning_step'] >= 100:
        if _skip_collate_stats['skipped_samples'] > 0:
            avg_skip = _skip_collate_stats['skipped_samples'] / _skip_collate_stats['total_batches']
            print(f"[Batch Stats] After {_skip_collate_stats['total_batches']} batches: "
                  f"Average {avg_skip:.2f} samples skipped per batch. "
                  f"This causes variable batch sizes. "
                  f"Current batch: {original_size} -> {len(clean_batch)}")
            _skip_collate_stats['last_warning_step'] = _skip_collate_stats['total_batches']

    return default_collate(clean_batch)



def prepare_TUAB_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_KaggleERN_dataset(root, data_key='signal', label_key='label', select_channel_idx=None):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    # select_channel_idx: [KaggleERN 通道修复] 见 kaggleern_select_channel_idx，
    # None 时行为完全不变。
    train_dataset = KaggleERNLoader(os.path.join(root, "train"), train_files, data_key=data_key, label_key=label_key, select_channel_idx=select_channel_idx)
    test_dataset = KaggleERNLoader(os.path.join(root, "test"), test_files, data_key=data_key, label_key=label_key, select_channel_idx=select_channel_idx)
    val_dataset = KaggleERNLoader(os.path.join(root, "val"), val_files, data_key=data_key, label_key=label_key, select_channel_idx=select_channel_idx)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_Motor_dataset(root, data_key='signal', label_key='label'):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = MotorLoader(os.path.join(root, "train"), train_files, data_key=data_key, label_key=label_key)
    test_dataset = MotorLoader(os.path.join(root, "test"), test_files, data_key=data_key, label_key=label_key)
    val_dataset = MotorLoader(os.path.join(root, "val"), val_files, data_key=data_key, label_key=label_key)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_Motor_dataset_subject_independent(root, test_subject, val_subject, data_key='signal', label_key='label'):
    """[LOSO] 20折 subject-independent 划分：test = test_subject, val = val_subject,
    train = 其余全部受试者。与 cbramod_finetune/datasets/motortask_dataset.py、
    biot_finetune 的 Motion 划分使用同一批底层 AllSubjects_Epochs pickle 文件、
    同一套受试者提取规则，保证同一折在不同模型间的 train/val/test 受试者集合完全一致。
    """
    subject_to_files = list_motor_files_by_subject(root)
    subjects = sorted(subject_to_files.keys(), key=lambda s: int(s[3:]))
    if len(subjects) < 3:
        raise ValueError(f"Need at least 3 subjects for subject-independent split, got {len(subjects)}: {subjects}")
    for s in (test_subject, val_subject):
        if s not in subject_to_files:
            raise ValueError(f"Subject {s!r} not found among {subjects}")
    if test_subject == val_subject:
        raise ValueError("test_subject and val_subject must be different")

    train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]
    val_subjects = [val_subject]
    test_subjects = [test_subject]

    def gather(subj_list):
        files = []
        for s in subj_list:
            files.extend(subject_to_files[s])
        return files

    train_files = gather(train_subjects)
    val_files = gather(val_subjects)
    test_files = gather(test_subjects)

    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(train_files)

    print("=" * 70)
    print(f"[split_mode=subject_independent] test={test_subject} val={val_subject} "
          f"train={len(train_subjects)} subjects")
    print(f"  All subjects ({len(subjects)}): {subjects}")
    print(f"  file counts: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print("=" * 70)

    # files 已经是相对 cwd 的完整路径（list_motor_files_by_subject 已经拼过 split 目录），
    # 所以这里 root 传空字符串，os.path.join("", full_path) 等价于 full_path 本身。
    train_dataset = MotorLoader("", train_files, data_key=data_key, label_key=label_key)
    test_dataset = MotorLoader("", test_files, data_key=data_key, label_key=label_key)
    val_dataset = MotorLoader("", val_files, data_key=data_key, label_key=label_key)
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset, val_dataset


def prepare_Stress_dataset_subject_independent(root, test_subject, val_subject, data_key='X', label_key='y'):
    """[LOSO] 17折 subject-independent 划分：test = test_subject, val = val_subject,
    train = 其余全部受试者。与 cbramod_finetune/neurolm_finetune 的 Stress LOSO 划分
    使用同一批底层 preprocessing/stress_preprocess.ipynb 生成的 pickle 文件、
    同一套受试者提取规则，保证同一折在不同模型间的 train/val/test 受试者集合完全一致。
    """
    subject_to_files = list_stress_files_by_subject(root)
    subjects = sorted(subject_to_files.keys(), key=lambda s: int(s[3:]))
    if len(subjects) < 3:
        raise ValueError(f"Need at least 3 subjects for subject-independent split, got {len(subjects)}: {subjects}")
    for s in (test_subject, val_subject):
        if s not in subject_to_files:
            raise ValueError(f"Subject {s!r} not found among {subjects}")
    if test_subject == val_subject:
        raise ValueError("test_subject and val_subject must be different")

    train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]
    val_subjects = [val_subject]
    test_subjects = [test_subject]

    def gather(subj_list):
        files = []
        for s in subj_list:
            files.extend(subject_to_files[s])
        return files

    train_files = gather(train_subjects)
    val_files = gather(val_subjects)
    test_files = gather(test_subjects)

    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(train_files)

    print("=" * 70)
    print(f"[split_mode=subject_independent] test={test_subject} val={val_subject} "
          f"train={len(train_subjects)} subjects")
    print(f"  All subjects ({len(subjects)}): {subjects}")
    print(f"  file counts: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print("=" * 70)

    # files 已经是相对 cwd 的完整路径（list_stress_files_by_subject 已经拼过 split 目录），
    # 所以这里 root 传空字符串，os.path.join("", full_path) 等价于 full_path 本身。
    train_dataset = StressLoader("", train_files, data_key=data_key, label_key=label_key)
    test_dataset = StressLoader("", test_files, data_key=data_key, label_key=label_key)
    val_dataset = StressLoader("", val_files, data_key=data_key, label_key=label_key)
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset, val_dataset


def _collect_pickle_files(root_dir):
    """
    递归收集目录下所有.pkl和.pickle文件
    返回相对于root_dir的文件路径列表
    """
    pickle_files = []
    root_dir = os.path.abspath(root_dir)
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pkl') or file.endswith('.pickle'):
                full_path = os.path.join(root, file)
                # 获取相对于root_dir的相对路径
                rel_path = os.path.relpath(full_path, root_dir)
                pickle_files.append(rel_path)
    return pickle_files

def prepare_Seed_dataset(root, data_key='signal', label_key='label'):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    # 递归收集所有pickle文件（包括子文件夹中的）
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    
    train_files = _collect_pickle_files(train_dir)
    val_files = _collect_pickle_files(val_dir)
    test_files = _collect_pickle_files(test_dir)
    
    np.random.shuffle(train_files)
    
    print(f"Found {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")

    # prepare training and test data loader
    train_dataset = SeedLoader(train_dir, train_files, data_key=data_key, label_key=label_key, expected_channels=62)
    test_dataset = SeedLoader(test_dir, test_files, data_key=data_key, label_key=label_key, expected_channels=62)
    val_dataset = SeedLoader(val_dir, val_files, data_key=data_key, label_key=label_key, expected_channels=62)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset, val_dataset

def prepare_Sleep_dataset(root, data_key='signal', label_key='label'):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = SleepLoader(os.path.join(root, "train"), train_files, data_key=data_key, label_key=label_key, expected_channels=6)
    test_dataset = SleepLoader(os.path.join(root, "test"), test_files, data_key=data_key, label_key=label_key, expected_channels=6)
    val_dataset = SleepLoader(os.path.join(root, "val"), val_files, data_key=data_key, label_key=label_key, expected_channels=6)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        # [LOSO] Stress 某一折的 val/test 受试者可能只有单一类别（17 个受试者里有 11 个
        # 只做过 increase 或只做过 normal），此时 roc_auc/pr_auc 在数学上未定义
        # （sklearn 会直接 raise ValueError），但 accuracy/balanced_accuracy 不受影响、
        # 完全可以正常算。旧代码在这种情况下把四个指标全部硬编码成 0.0（不是没算出
        # 来，是压根没调用 sklearn），导致 loso_mode 用 val balanced_accuracy 选
        # best epoch 时，只要某折的 val 受试者是单类别，每个 epoch 都会读到硬编码的
        # 0.0，best_epoch 就会永远停在第 1 个满足 0.0 > -1.0 的 epoch，跟模型实际
        # 训练效果毫无关系。这里改成只对不可定义的 roc_auc/pr_auc 返回 NaN，
        # accuracy/balanced_accuracy 照常计算 -- 跟 cbramod_finetune/finetune_evaluator.py
        # 的 get_metrics_for_binaryclass 是同一套处理方式。
        degenerate = sum(target) * (len(target) - sum(target)) == 0
        undefined_metrics = {'roc_auc', 'pr_auc'} if degenerate else set()
        results = binary_metrics_fn(
            target,
            output,
            metrics=[m for m in metrics if m not in undefined_metrics],
            threshold=threshold,
        )
        for m in undefined_metrics:
            if m in metrics:
                results[m] = float('nan')
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results


def extract_video_index(epoch_id):
    """从 epoch_id 中提取 video_index"""
    import re
    match = re.search(r'video_index_(\d+)_chunk', epoch_id)
    if match:
        return int(match.group(1))
    return None


def extract_subject_id(epoch_id):
    """从 epoch_id 中提取 subject_id"""
    import re
    match = re.search(r'subject_(\d+)_', epoch_id)
    if match:
        return int(match.group(1))
    return None


def compute_video_level_metrics(pred_classes, true_classes, epoch_ids, is_binary):
    """
    计算视频级和subject级的评估指标
    
    参数:
        pred_classes: 预测类别数组 (n_samples,)
        true_classes: 真实类别数组 (n_samples,)
        epoch_ids: epoch_id列表 (n_samples,)
        is_binary: 是否为二分类任务
    
    返回:
        包含视频级和subject级指标的字典
    """
    from collections import defaultdict, Counter
    import numpy as np
    
    # 按video_index分组
    video_groups = defaultdict(list)
    for i, epoch_id in enumerate(epoch_ids):
        video_index = extract_video_index(epoch_id)
        if video_index is not None:
            video_groups[video_index].append({
                'pred': pred_classes[i],
                'true': true_classes[i],
                'epoch_id': epoch_id
            })
    
    # 对每个视频进行投票
    video_results = []
    for video_index, chunks in sorted(video_groups.items()):
        if len(chunks) == 0:
            continue
        
        # 统计每个类别的投票数
        preds = [ch['pred'] for ch in chunks]
        true_label = chunks[0]['true']  # 同一视频的true label应该相同
        
        vote_counts = Counter(preds)
        max_votes = max(vote_counts.values())
        
        # 找出得票最多的类别
        winners = [label for label, count in vote_counts.items() if count == max_votes]
        
        if len(winners) == 1:
            # 唯一胜者
            pred_label = winners[0]
            is_correct = 1.0 if pred_label == true_label else 0.0
        else:
            # 平票情况：如果真实标签在平票候选中，算0.5
            if true_label in winners:
                is_correct = 0.5
            else:
                is_correct = 0.0
        
        video_results.append({
            'video_index': video_index,
            'pred_label': winners[0] if len(winners) == 1 else winners,
            'true_label': true_label,
            'is_correct': is_correct,
            'num_chunks': len(chunks)
        })
    
    # 计算视频级准确率
    if len(video_results) > 0:
        video_accuracy = sum(v['is_correct'] for v in video_results) / len(video_results)
    else:
        video_accuracy = 0.0
    
    # 按subject分组计算准确率
    subject_groups = defaultdict(list)
    for result in video_results:
        # 从对应的chunks中提取subject_id
        chunks = video_groups[result['video_index']]
        if chunks and len(chunks) > 0:
            subject_id = extract_subject_id(chunks[0]['epoch_id'])
            if subject_id is not None:
                subject_groups[subject_id].append(result)
    
    # 计算每个subject的准确率
    subject_accuracies = {}
    for subject_id, videos in subject_groups.items():
        if len(videos) > 0:
            total_correct = sum(v['is_correct'] for v in videos)
            subject_accuracies[f'subject_{subject_id}_accuracy'] = total_correct / len(videos)
            subject_accuracies[f'subject_{subject_id}_num_videos'] = len(videos)
    
    # 计算平均subject准确率
    if len(subject_accuracies) > 0:
        subject_acc_values = [v for k, v in subject_accuracies.items() if k.endswith('_accuracy')]
        mean_subject_accuracy = np.mean(subject_acc_values) if subject_acc_values else 0.0
    else:
        mean_subject_accuracy = 0.0
    
    return {
        'video_level_accuracy': video_accuracy,
        'video_level_num_videos': len(video_results),
        'mean_subject_accuracy': mean_subject_accuracy,
        'num_subjects': len(subject_groups),
        **subject_accuracies
    }
