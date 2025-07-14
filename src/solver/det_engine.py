"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import BaseEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, IoU
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from torchmetrics.segmentation import MeanIoU
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from itertools import cycle

import cv2
import random
import numpy as np

# torch.autograd.set_detect_anomaly(True)

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

class ImageWriter():
    def __init__(self, n_output):
        self.n_output = n_output
        # def color map by n classes
        self.color_map = [(194, 118, 1), (150, 150, 200), (0, 200, 200), (231, 73, 30), (31, 252, 20), (39, 13, 184), (187, 3, 208), (103, 1, 68)]
    def __get_random_color__(self):
        rgbl = [0, 0, 0]
        while ((math.fabs(min(rgbl) - max(rgbl)) < 75)):
            rgbl = []
            for i in range(3):
                channel_value = np.random.randint(0, 255)
                rgbl.append(channel_value)
        return tuple(rgbl)

    def __euclidean_color_dist__(self, color1, color2):
        r1 = color1[0]
        g1 = color1[1]
        b1 = color1[2]
        r2 = color2[0]
        g2 = color2[1]
        b2 = color2[2]
        d = math.sqrt(2 * math.pow((r1 - r2), 2) + 4 * math.pow((g1 - g2), 2) + 3 * math.pow((b1 - b2), 2))

        return d

    def __check_color_vis__(self, color, map, restr):
        if (len(map) < 1):
            return True
        for c in map:
            d = self.__euclidean_color_dist__(c, color)
            if d < restr:
                return False
        return True

    def __gen_color_map__(self, n_color):
        color_map = []
        for i in range(n_color):
            rnd_color = self.__get_random_color__()
            while(not self.__check_color_vis__(rnd_color, color_map, 250)):
                rnd_color = self.__get_random_color__()
            color_map.append(rnd_color)

        self.color_map = color_map
        return color_map

imageWriter = ImageWriter(6)

def check_numerics(tensor_dict: dict, name="outputs"):
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"NaN or Inf detected in {name}['{k}']")
                return True
    return False

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loaders: Iterable[Iterable],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 50)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    world_size = 1
    if dist_utils.is_dist_available_and_initialized():
        world_size = dist_utils.get_world_size()

    # Создаем итераторы и определяем минимальную длину для одной эпохи
    data_iters = [iter(dl) for dl in data_loaders]
    max_len = 50 #max(len(dl) for dl in data_loaders) * world_size

    for i, _ in enumerate(metric_logger.log_every(range(max_len), print_freq, header)):
        samples_list = []
        targets_list = []

        # Обрабатываем каждый data_loader
        for idx, (dl, it) in enumerate(zip(data_loaders, data_iters)):
            try:
                samples, targets = next(it)
            except StopIteration:
                data_iters[idx] = iter(dl)
                samples, targets = next(data_iters[idx])
            
            samples_list.append(samples)
            targets_list.extend(targets)

        # Объединяем батчи и отправляем на устройство
        samples = torch.cat(samples_list, dim=0).to(device)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets_list
        ]

        global_step = epoch * max_len + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=max_len)

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets, mode=1) # 0 for det only and 2 for shared
                

            if check_numerics(outputs):
                print("NaNs detected, saving model state")
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "epoch": epoch,
                    "step": global_step,
                }
                dist_utils.save_on_master(state, "./nan_checkpoint.pth")
                raise RuntimeError("NaNs encountered in model outputs")

            # if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
            #     print(outputs['pred_boxes'])
            #     state = model.state_dict()
            #     new_state = {}
            #     for key, value in model.state_dict().items():
            #         # Replace 'module' with 'model' in each key
            #         new_key = key.replace("module.", "")
            #         # Add the updated key-value pair to the state dictionary
            #         state[new_key] = value
            #     new_state["model"] = state
            #     dist_utils.save_on_master(new_state, "./NaN.pth")
                
                
            with torch.autocast(device_type=str(device), enabled=False): # enabled=False
                loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # set_to_none=True

            # with torch.no_grad():
            #     torch.cuda.empty_cache()

        else:
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter still requires grad: {name}")

            outputs = model(samples, targets=targets, mode=1)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_iou(pred_mask, target_mask, num_classes):
    """
    Вычисление IoU для каждой из классов и возвращение среднего значения.

    :param pred_mask: Предсказанная маска (H x W).
    :param target_mask: Истинная маска (H x W).
    :param num_classes: Количество классов.
    :return: Среднее значение IoU по всем классам.
    """
    ious_scores = torch.zeros(num_classes)
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        target_cls = (target_mask == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union > 0:
            iou_scores.append((intersection / union).item())
            ious_scores[cls] = (intersection / union).item()
    return torch.mean(torch.tensor(iou_scores)) if iou_scores else 0.0, ious_scores
    # return torch.mean(torch.tensor(iou_scores)) if iou_scores else 0.0

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    evaluator: BaseEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    evaluator.reset()
    
    ##############
    from ..data.dataset.segmetric import SegMetric
    segmetric = SegMetric(6, None)
    ##############

    metric_logger = MetricLogger(delimiter="  ")
    header = f"Validation ({getattr(data_loader, '_meta', {}).get('name', 'unknown')})"

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        global_step = epoch * len(data_loader) + i

        # if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
        #     save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)

            # Если нужен постпроцессинг
            if postprocessor:
                _, _, H_, W_ = samples.shape
                # Получаем оригинальные размеры из samples (обычно: (N, C, H, W))
                orig_target_sizes = torch.tensor([W_, H_], device=samples.device)
                outputs, targets_tensor = postprocessor(outputs, orig_target_sizes, targets)
            else:
                # Попробовать собрать тензор вручную
                targets_tensor = targets

            evaluator.update(outputs, targets_tensor)

            # TODO: временное решение для валидации сегментации, необходимо переписать !!!
            if (isinstance(outputs, torch.Tensor)):
                segmetric.update(outputs.cpu().detach().numpy(), targets_tensor.cpu().detach().numpy())
    # Синхронизация для DDP
    metric_logger.synchronize_between_processes()

    segmetric.write_results()

    return evaluator