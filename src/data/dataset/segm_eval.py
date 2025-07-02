from ...core import register

from .evaluator import Evaluator
from .segmetric import SegMetric

__all__ = [
    "SegmentationEvaluator",
]


@register()
class SegmentationEvaluator(Evaluator):
    def __init__(self, num_classes, metric='IoU', summary_writer=None):
        self.metric = SegMetric(outputs=num_classes, summary_writer=summary_writer)

    def update(self, labels_preds):
        labels, preds = labels_preds
        self.metric.update(labels, preds)

    def synchronize_between_processes(self):
        # Если используется DDP — реализовать или оставить пустым
        pass

    def accumulate(self):
        # Здесь метрика считает по ходу update, но можно добавить итоговую обработку
        pass

    def summarize(self):
        results = self.metric.get()
        print("Segmentation Metrics:")
        for metric in results:
            print(f"{metric[0]}: {metric[1]}")