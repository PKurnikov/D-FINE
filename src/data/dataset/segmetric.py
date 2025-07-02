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
                if (self.statCollector[i].fp + self.statCollector[i].tp) > 0:
                    self.statCollector[i].precision = self.statCollector[i].tp * 1. / (self.statCollector[i].fp + self.statCollector[i].tp)
                else:
                    self.statCollector[i].precision = 0.0
            else:
                self.statCollector[i].precision = 0.0

    def recall(self):
        for c in self.statCollector:
            if (c.gtp != 0):
                if (c.fn + c.tp) > 0:
                    c.recall = c.tp * 1. / (c.fn + c.tp)
                else:
                    c.recall = 0.0
            else:
                c.recall = 0.0

    def iou(self):
        for c in self.statCollector:
            if (c.gtp != 0):
                if (c.fn + c.tp + c.fp) > 0:
                    c.iou = c.tp * 1. / (c.fn + c.tp + c.fp)
                else:
                    c.iou = 0.0
            else:
                c.iou = 0.0

    def weightedIoU(self):
        weights = []
        for i in self.statCollector:
            if (i.gtp != 0):
                weights.append(i.avgGtTp / i.avgTp)
            else:
                weights.append(0.0)
        for i in range(self.statCollector.__len__()):
            if (self.statCollector[i].gtp != 0):
                if (self.statCollector[i].fn * weights[i] + self.statCollector[i].tp * weights[i] + self.statCollector[i].fp) > 0:
                    self.statCollector[i].wIou = self.statCollector[i].tp * 1.0 * weights[i] / (self.statCollector[i].fn * weights[i] + self.statCollector[i].tp * weights[i] + self.statCollector[i].fp)
                else:
                    self.statCollector[i].wIou = 0.0
            else:
                self.statCollector[i].wIou = 0.0

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

    def get(self, types=['precision', 'recall', 'iou', 'wIou']):
        metric_list = []
        for i, c in enumerate(self.statCollector):
            for n, v in c.__dict__.items():
                if n in types:
                    name_metric = '{0}_of_class_{1}'.format(n, str(i))
                    metric = (name_metric, v)
                    metric_list.append(metric)
        return metric_list

    def write_results(self, step=-1):
        metrics = self.get(['iou'])
        for m in metrics:
            tag, value = m
            print(f"  {tag}: {value:.4f}")  # <-- Печать в консоль
            # self.writer.add_scalar(tag=tag, scalar_value=value,
            #                        global_step=step)