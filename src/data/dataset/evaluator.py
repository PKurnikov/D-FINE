from ...core import register
from ...misc import dist_utils
from torchmetrics import Metric
import torch

__all__ = [
    "SegmentationEvaluator",
    "DetectionEvaluator"
]

@register()
class BaseEvaluator:
    def __init__(self, device=None):
        self.device = device
        self.metric = Metric  # torchmetrics.Metric или список Metric

    def update(self, preds, targets):
        if isinstance(self.metric, Metric):
            self.metric.update(preds, targets)
        elif isinstance(self.metric, (list, tuple)):
            for m in self.metric:
                m.update(preds, targets)

    def compute(self):
        if isinstance(self.metric, Metric):
            return self.metric.compute()
        elif isinstance(self.metric, (list, tuple)):
            return {type(m).__name__: m.compute() for m in self.metric}

    def reset(self):
        if isinstance(self.metric, Metric):
            self.metric.reset()
        elif isinstance(self.metric, (list, tuple)):
            for m in self.metric:
                m.reset()

    def summary(self) -> str:
            result = self.compute()

            def _format_result(name, val):
                if isinstance(val, torch.Tensor):
                    if val.ndim == 0:
                        return f"{name}: {val.item():.4f}"
                    elif val.ndim == 1:
                        return f"{name} per class:\n" + "\n".join(
                            [f"  Class {i}: {v.item():.4f}" for i, v in enumerate(val)]
                        )
                    else:
                        return f"{name}: tensor with shape {val.shape}"
                elif isinstance(val, dict):
                    return f"{name}:\n" + "\n".join(
                        [f"  {k}: {v:.4f}" if isinstance(v, (float, int)) else f"  {k}: {v}" for k, v in val.items()]
                    )
                else:
                    return f"{name}: {val}"

            if isinstance(result, (list, tuple)):
                results = {type(m).__name__: m.compute() for m in self.metric}
                return "\n".join([f"{k}: {v}" for k, v in results.items()])

            # Single metric
            if isinstance(result, (torch.Tensor, dict, float)):
                return _format_result(type(self.metric).__name__, result)

            # Multiple metrics
            elif isinstance(result, dict):
                return "\n".join([_format_result(name, val) for name, val in result.items()])

            return f"{type(self.metric).__name__}: Unsupported result format"

from torchmetrics.segmentation import MeanIoU, DiceScore, GeneralizedDiceScore, HausdorffDistance
from torchmetrics.detection import MeanAveragePrecision

@register()
class SegmentationEvaluator(BaseEvaluator):
    def __init__(self, metric, num_classes, input_format, per_class, device=None):
        super().__init__(device)
        if metric == 'MeanIoU':
            self.metric = MeanIoU(num_classes=num_classes, input_format=input_format, per_class=per_class).to(device)
        if metric == 'DiceScore':
            self.metric = DiceScore(num_classes=num_classes, input_format=input_format).to(device)
        if metric == 'GeneralizedDiceScore':
            self.metric = GeneralizedDiceScore(num_classes=num_classes, input_format=input_format).to(device)

@register()
class DetectionEvaluator(BaseEvaluator):
    def __init__(self, iou_types=None, iou_thresholds=None, box_format='xyxy', device=None):
        super().__init__(device)

        self.metric = MeanAveragePrecision(iou_type=iou_types, iou_thresholds=iou_thresholds, box_format=box_format).to(device)

    def compute(self):
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        return self.metric.compute()

    def reset(self):
        self.metric.reset()