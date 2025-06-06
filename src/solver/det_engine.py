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

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, IoU
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from torchmetrics.segmentation import MeanIoU

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

#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/03/10
# @Author  : kurnikov_pavel
# @Contact : thetwonames@gmail.com
# @File    : SegMetric.py

class UnitStat():
    def __init__(self):
        self.tp = 0
        self.tpW = 0
        self.fp = 0
        self.fn = 0
        self.fnW = 0
        self.gtp = 0
        self.avgTp = 0.
        self.avgGtTp = 0.
        self.precision = 0.
        self.recall = 0.
        self.iou = 0.
        self.wIou = 0.

# custom eval metric
class SegMetric():
    """CalculSegMetricate metrics for Seg training """
    def __init__(self, outputs, summary_writer):
        self.outputs = outputs
        self.writer = summary_writer
        self.statCollector = [UnitStat() for i in range(self.outputs)]
        self.sample_size = 0
        super(SegMetric, self).__init__()

    def precision(self):
        for i in range(self.statCollector.__len__()):
            if (self.statCollector[i].gtp != 0):
                self.statCollector[i].precision = self.statCollector[i].tp * 1. / (self.statCollector[i].fp + self.statCollector[i].tp)
            else:
                self.statCollector[i].precision = 0

    def recall(self):
        for c in self.statCollector:
            if (c.gtp != 0):
                c.recall = c.tp * 1. / (c.fn + c.tp)
            else:
                c.recall = 0

    def iou(self):
        for c in self.statCollector:
            if (c.gtp != 0):
                c.iou = c.tp * 1. / (c.fn + c.tp + c.fp)
            else:
                c.iou = 0

    def weightedIoU(self):
        weights = []
        for i in self.statCollector:
            if (i.gtp != 0):
                weights.append(i.avgGtTp / i.avgTp)
            else:
                weights.append(0.0)
        for i in range(self.statCollector.__len__()):
            if (self.statCollector[i].gtp != 0):
                self.statCollector[i].wIou = self.statCollector[i].tp * 1.0 * weights[i] / (self.statCollector[i].fn * weights[i] + self.statCollector[i].tp * weights[i] + self.statCollector[i].fp)
            else:
                self.statCollector[i].wIou = 0

    def reset(self):
        self.sample_size = 0
        self.statCollector = [UnitStat() for i in range(self.outputs)]

    def update(self, labels, preds):
        # labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            # if pred_label.shape != label.shape:
            #     pred_label = mx.ndarray.argmax(pred_label, axis=1)

            # updating examples size
            self.sample_size += 1

            dcmask = (label == 255) * 255
            pred_label = pred_label * (label != 255) + dcmask

            for c in range(self.outputs):
                diff = (pred_label == c).astype(int) - (label == c).astype(int)
                fp = (diff == 1).sum()
                fn = (diff == -1).sum()
                tp = (((label == c).astype(int) - (diff == 1).astype(int)) == 1).sum()
                gtp = (pred_label == c).sum()

                self.statCollector[c].fn += fn#.asscalar()
                self.statCollector[c].fp += fp#.asscalar()
                self.statCollector[c].tp += tp#.asscalar()
                self.statCollector[c].gtp += gtp#.asscalar()

                self.statCollector[c].avgGtTp = self.statCollector[c].gtp * 1. / self.sample_size
                self.statCollector[c].avgTp = self.statCollector[c].tp * 1. / self.sample_size

            self.precision()
            self.recall()
            self.iou()
            self.weightedIoU()

    def get(self):
        metric_list = []
        for i, c in enumerate(self.statCollector):
            for n, v in c.__dict__.items():
                if n in ['precision', 'recall', 'iou', 'wIou']:
                    name_metric = '{0}_of_class_{1}'.format(n, str(i))
                    metric = (name_metric, v)
                    metric_list.append(metric)
        return metric_list

    def write_results(self, step):
        metrics = self.get()
        for m in metrics:
            self.writer.add_scalar(tag=m[0], scalar_value=m[1],
                                   global_step=(step))

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

    # data_loader = data_loaders[0]
    # seg_data_loader = data_loaders[1]
    # dataloader_iterator = iter(seg_data_loader)
    # dataloader_iterator_det = iter(data_loader)

    world_size = 1
    if dist_utils.is_dist_available_and_initialized():
        world_size = dist_utils.get_world_size()

    # Создаем итераторы и определяем минимальную длину для одной эпохи
    data_iters = [iter(dl) for dl in data_loaders]
    max_len = max(len(dl) for dl in data_loaders) * world_size

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

    # for i, i_ in enumerate(metric_logger.log_every(list(range(4374)), print_freq, header)):
    #     try:
    #         samples1, targets1 = next(dataloader_iterator_det)
    #     except StopIteration:
    #         dataloader_iterator_det = iter(data_loader)
    #         samples1, targets1 = next(dataloader_iterator_det)

    #     try:
    #         samples2, targets2 = next(dataloader_iterator)
    #     except StopIteration:
    #         dataloader_iterator = iter(seg_data_loader)
    #         samples2, targets2 = next(dataloader_iterator)
           
        # samples = torch.cat((samples1, samples2), dim=0)
        # samples = samples.to(device)
        
        # targets = targets1 + targets2
        
        # targets = [
        #     {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        #     for t in targets
        # ]

        global_step = epoch * max_len + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=max_len)

        # global_step = epoch * len(data_loader) + i
        # metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        # if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
        #     save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets, det_mode=2) # 0 for det only and 2 for shared
                
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")
                
                
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
            outputs = model(samples, targets=targets)
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
    data_loaders,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    iou_scores = []  # Для хранения IoU по всем изображениям
    iou_scores_tensor = []
    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    data_loader = data_loaders[0]
    data_loader2 = data_loaders[1]

    for i, (samples1, targets1) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples1, targets1, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples1 = samples1.to(device)
        targets1 = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets1]
        outputs_det = model(samples1, None, 0)
        # # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets1], dim=0)
        results = postprocessor(outputs_det, orig_target_sizes)
        
        res = {target['image_id'].item(): output for target, output in zip(targets1, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets1, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples1[idx].shape[-1], samples1[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )


    from ..metrics.iou import IoU
    num_classes = 6 # 91
    metric = IoU(num_classes) # 91
    segMetric = SegMetric(outputs=num_classes, summary_writer=None)
    
    for samples2, targets2 in metric_logger.log_every(data_loader2, 10, header):            
        samples2 = samples2.to(device)
        targets2 = [{k: v.to(device) for k, v in t.items()} for t in targets2]
        
        outputs_seg = model(samples2, None, 1)

        # Расчет IoU для семантической сегментации
        if 'pred_sgm_head_2' in outputs_seg:
            pred_masks = torch.nn.functional.interpolate(outputs_seg['pred_sgm_head_2'], mode="bilinear", size=(576, 1024), align_corners=True)
            pred_masks = torch.softmax(pred_masks, dim=1)  # B x H x W
            
            cnt = 0
            for pred_mask, target in zip(pred_masks, targets2):
                if 'masks' in target and target['masks'].numel() > 0:
                    segMetric.update([target['masks'][0].data.cpu().numpy()], [torch.argmax(pred_mask, dim=0).data.cpu().numpy()])
                    iou, ious = compute_iou(torch.argmax(pred_mask, dim=0), target['masks'], num_classes=num_classes)
                    iou_scores.append(iou)
                    iou_scores_tensor.append(ious)
                    # cv2.imwrite(str(random.randint(0, 100))+".png", torch.argmax(pred_mask, dim=0).cpu().detach().numpy())
                  
                ### visualisation ###
                src = samples2[cnt].data.cpu().numpy().transpose(1, 2, 0)
                cnt += 1
                color_mask = np.zeros_like(src, dtype=int)
                for j, color in enumerate(imageWriter.color_map):
                    temp = np.ones_like(src, dtype=int)
                    temp = np.multiply(temp, color)
                    for c in range(3):
                        np.copyto(color_mask[:,:,c], temp[:,:,c], where = [torch.argmax(pred_mask, dim=0).data.cpu().numpy() == j][0])
                src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                src = (src * np.array([255., 255., 255.]))
                background = cv2.addWeighted(src, 0.99, color_mask.astype(float), 0.15, 0.0, 128)
                background = cv2.resize(background, (1280, 720))
                cv2.imwrite("vis/" + str(random.randint(0, 100)) + '.png', background) 
                ### visualisation ###


        # # with torch.autocast(device_type=str(device)):
        # #     outputs = model(samples)

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    # Вывод средней IoU для семантической сегментации
    if iou_scores:
        mean_iou = torch.mean(torch.tensor(iou_scores))
        print(f"Mean IoU for semantic segmentation: {mean_iou:.4f}")

    for i in range(len(segMetric.statCollector)):
        print('IOU of class {} = {}'.format(i, segMetric.statCollector[i].iou))

    return stats, coco_evaluator
