# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange
import numpy as np
def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # 每个epoch只打印一次
    print_freq = num_training_steps_per_epoch if num_training_steps_per_epoch else float('inf')

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 处理可能包含epoch_id的情况（Seed数据集）
        if len(batch_data) == 3:
            samples, targets, _ = batch_data  # 忽略epoch_id（训练时不需要）
        else:
            samples, targets = batch_data
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        # Reshape data for model input: [B, N, T] -> [B, N, A, 200], A = T // 200
        # 通用写法，按 T 自动算出 patch 数 A，覆盖当前所有任务：
        #   Motor T=200 (A=1) / KaggleERN T=600 (A=3) / Seed T=800 (A=4) /
        #   Stress T=1000 (A=5) / Sleep T=6000 (A=30)
        # 白名单校验：T 不在已知任务时长范围内就直接报错，防止前处理出问题时
        # 静默生成一个"能被200整除但语义不对"的长度被悄悄接受。
        _VALID_T = {200, 600, 800, 1000, 6000}
        if len(samples.shape) == 3:  # [B, N, T]
            B, N, T = samples.shape
            if T not in _VALID_T:
                raise ValueError(
                    f"Unexpected time dimension T={T}; expected one of {sorted(_VALID_T)} "
                    f"(Motor=200, KaggleERN=600, Seed=800, Stress=1000, Sleep=6000). "
                    f"This usually means the preprocessing pipeline produced an unexpected window length."
                )
            samples = samples.view(B, N, T // 200, 200)
        elif len(samples.shape) == 4:  # Already [B, N, A, T]
            pass  # Already in correct format
        else:
            raise ValueError(f"Unexpected input shape: {samples.shape}")
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            # 根据设备类型选择 autocast
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)
            elif device.type == 'mps':
                # MPS 使用 CPU autocast 或者不使用
                with torch.amp.autocast(device_type='cpu', dtype=torch.float16):
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)
            else:
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq  # 梯度累积：将loss除以update_freq，这样累积后的平均loss是正确的
            should_update = (data_iter_step + 1) % update_freq == 0
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=should_update)
            if should_update:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            # 安全地获取 loss scale 值
            try:
                loss_scale_value = loss_scaler._scaler.get_scale()
            except (KeyError, AttributeError):
                # 如果 get_scale() 不可用，尝试从 state_dict 获取
                scaler_state = loss_scaler.state_dict()
                loss_scale_value = scaler_state.get("scale", 1.0)

        # 只在 CUDA 设备上同步，MPS 和 CPU 不需要
        if device.type == 'cuda':
            torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    epoch_ids = []  # 收集epoch_id用于视频级投票
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        # 检查是否有epoch_id（Seed数据集返回3个值：X, Y, epoch_id）
        if len(batch) == 3:
            target = batch[1]  # batch[0]=EEG, batch[1]=targets, batch[2]=epoch_ids
            epoch_id_batch = batch[2]
            epoch_ids.extend(epoch_id_batch)
        else:
            target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        # Reshape data for model input: [B, N, T] -> [B, N, A, 200], A = T // 200
        # 通用写法，按 T 自动算出 patch 数 A，覆盖当前所有任务：
        #   Motor T=200 (A=1) / KaggleERN T=600 (A=3) / Seed T=800 (A=4) /
        #   Stress T=1000 (A=5) / Sleep T=6000 (A=30)
        # 白名单校验：T 不在已知任务时长范围内就直接报错，防止前处理出问题时
        # 静默生成一个"能被200整除但语义不对"的长度被悄悄接受。
        _VALID_T = {200, 600, 800, 1000, 6000}
        if len(EEG.shape) == 3:  # [B, N, T]
            B, N, T = EEG.shape
            if T not in _VALID_T:
                raise ValueError(
                    f"Unexpected time dimension T={T}; expected one of {sorted(_VALID_T)} "
                    f"(Motor=200, KaggleERN=600, Seed=800, Stress=1000, Sleep=6000). "
                    f"This usually means the preprocessing pipeline produced an unexpected window length."
                )
            EEG = EEG.view(B, N, T // 200, 200)
        elif len(EEG.shape) == 4:  # Already [B, N, A, T]
            pass  # Already in correct format
        else:
            raise ValueError(f"Unexpected input shape: {EEG.shape}")
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        
        # compute output
        # 根据设备类型选择 autocast
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                output = model(EEG, input_chans=input_chans)
                loss = criterion(output, target)
        elif device.type == 'mps':
            # MPS 使用 CPU autocast 或者不使用
            with torch.amp.autocast(device_type='cpu', dtype=torch.float16):
                output = model(EEG, input_chans=input_chans)
                loss = criterion(output, target)
        else:
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    
    # 获取预测类别和真实类别（用于混淆矩阵和视频级投票）
    if len(pred.shape) > 1:
        pred_classes = np.argmax(pred, axis=1)
    else:
        pred_classes = (pred > 0.5).astype(int)
    
    # 确保true是整数类别
    if len(true.shape) > 1:
        true_classes = np.argmax(true, axis=1) if true.shape[1] > 1 else true.flatten().astype(int)
    else:
        true_classes = true.astype(int)
        
    # 计算混淆矩阵（仅对多分类任务）
    if not is_binary:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_classes, pred_classes)
        # 将混淆矩阵转换为列表以便JSON序列化
        ret['confusion_matrix'] = cm.tolist()
    
    # 如果收集到了epoch_id，进行视频级投票评估（仅对Seed数据集）
    if len(epoch_ids) > 0 and len(epoch_ids) == len(pred_classes):
        video_level_metrics = utils.compute_video_level_metrics(
            pred_classes, true_classes, epoch_ids, is_binary
        )
        ret.update(video_level_metrics)
    
    return ret