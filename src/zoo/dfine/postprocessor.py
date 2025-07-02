"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out

@register()
class SharedPostProcessor(nn.Module):
    def __init__(
        self,
        heads: dict,
        deploy_mode: bool = False,
    ):
        super().__init__()
        self.deploy_mode = deploy_mode

        self.detector_enabled = False
        self.seg_enabled = False

        # Order of outputs (in deploy mode): based on config ordering
        self.return_order = []

        for head_type, cfg in heads.items():
            if not cfg.get("enable", False):
                continue

            # ---- Detector config ----
            if head_type == "detector":
                self.detector_enabled = True
                d_cfg = heads["detector"]
                self.det_logits_key = d_cfg.get("logits_key", "pred_logits")
                self.det_boxes_key = d_cfg.get("boxes_key", "pred_boxes")
                self.det_num_classes = d_cfg.get("num_classes", 80)
                self.det_use_focal_loss = d_cfg.get("use_focal_loss", True)
                self.det_num_top_queries = d_cfg.get("num_top_queries", 300)
                self.det_remap_mscoco_category = d_cfg.get("remap_mscoco_category", False)
                self.in_fmt=d_cfg.get("in_fmt", 'cxcywh')
                self.out_fmt=d_cfg.get("out_fmt", 'xyxy')
                self.scale_factor=d_cfg.get("scale_factor", 1.0)

                self.return_order.append("detections")

            # ---- Segmentation config ----
            if head_type == "segmentation":
                self.seg_enabled = True
                s_cfg = heads["segmentation"]
                self.seg_outputs = s_cfg.get("outputs", [])
                self.seg_do_softmax = s_cfg.get("do_softmax", True)
                self.seg_do_argmax = s_cfg.get("do_argmax", False)
                self.seg_do_concat = s_cfg.get("do_concat", False)

                if self.seg_do_concat:
                    self.return_order.append("segmentations")
                else:
                    self.return_order.extend(self.seg_outputs)

    def forward(self, outputs, orig_target_sizes: torch.Tensor = None, targets=None):
        results = {}
        N = outputs[self.det_logits_key].shape[0] if self.detector_enabled else outputs[self.seg_outputs[0]].shape[0]

        ### ---- DETECTION PART ---- ###
        if self.detector_enabled:
            logits = outputs[self.det_logits_key]      # (N, num_queries, num_classes)
            boxes = outputs[self.det_boxes_key]        # (N, num_queries, 4)

            boxes = torchvision.ops.box_convert(boxes, in_fmt=self.in_fmt, out_fmt=self.out_fmt)
            boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)  # (N, num_queries, 4)

            if self.det_use_focal_loss:
                scores = torch.sigmoid(logits)
                scores, index = torch.topk(scores.flatten(1), self.det_num_top_queries, dim=-1)
                labels = index % self.det_num_classes
                index = index // self.det_num_classes
                boxes = boxes.gather(1, index.unsqueeze(-1).repeat(1, 1, 4))
            else:
                probs = F.softmax(logits, dim=-1)[:, :, :-1]
                scores, labels = probs.max(dim=-1)
                if scores.shape[1] > self.det_num_top_queries:
                    scores, index = torch.topk(scores, self.det_num_top_queries, dim=-1)
                    labels = labels.gather(1, index)
                    boxes = boxes.gather(1, index.unsqueeze(-1).repeat(1, 1, 4))

            if self.deploy_mode:
                detections = torch.cat([boxes, scores.unsqueeze(2), labels.unsqueeze(2)], dim=2)
                results["detections"] = detections[:, :self.det_num_top_queries, :]  # (N, 100, 6)
            else:
                detections = []
                for lab, box, sco in zip(labels, boxes, scores):
                    detections.append(dict(labels=lab, boxes=box, scores=sco))
                results["detections"] = detections

        ### ---- SEGMENTATION PART ---- ###
        if self.seg_enabled:
            seg_list = []
            seg_results = {}
            for head_name in self.seg_outputs:
                if head_name not in outputs:
                    continue
                seg = outputs[head_name]  # (N, C, H, W)

                if self.seg_do_softmax:
                    seg = F.softmax(seg, dim=1)
                if self.seg_do_argmax:
                    seg = torch.argmax(seg, dim=1).unsqueeze(0)  # (N, 1, H, W)

                seg_list.append(seg)
                seg_results[head_name] = seg

            if self.seg_do_concat:
                # Concat along channel dimension
                concat_seg = torch.cat(seg_list, dim=1)
                results["segmentations"] = concat_seg
            else:
                results.update(seg_results)

        ### ---- FINAL OUTPUT ---- ###
        if self.deploy_mode:
            return tuple(results[k] for k in self.return_order if k in results)
        else:
            return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self


# @register()
# class SharedPostProcessor(nn.Module):
#     __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

#     def __init__(
#         self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
#     ) -> None:
#         super().__init__()
#         self.use_focal_loss = use_focal_loss
#         self.num_top_queries = num_top_queries
#         self.num_classes = int(num_classes)
#         self.remap_mscoco_category = remap_mscoco_category
#         self.deploy_mode = False

#     def extra_repr(self) -> str:
#         return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

#     # def forward(self, outputs, orig_target_sizes):
#     def forward(self, outputs, orig_target_sizes, targets=None):
#         logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
#         # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         if 'pred_sgm_head_2' in outputs:
#             masks = torch.softmax(outputs['pred_sgm_head_2'], dim=1)
#             # return masks
        
#         logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
                
#         # else:
#         #     raise KeyError("Key 'pred_sgm_head_0' not found in outputs.")

#         bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') # xyxy
#         bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

#         if self.use_focal_loss:
#             scores = F.sigmoid(logits)
#             scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
#             # TODO for older tensorrt
#             # labels = index % self.num_classes
#             labels = mod(index, self.num_classes)
#             index = index // self.num_classes
#             boxes = bbox_pred.gather(
#                 dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
#             )

#         else:
#             scores = F.softmax(logits)[:, :, :-1]
#             scores, labels = scores.max(dim=-1)
#             if scores.shape[1] > self.num_top_queries:
#                 scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
#                 labels = torch.gather(labels, dim=1, index=index)
#                 boxes = torch.gather(
#                     boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
#                 )

#         # TODO for onnx export
#         if self.deploy_mode:
#             scores = scores.unsqueeze(2)
#             labels = labels.unsqueeze(2)
#             detections = torch.cat([boxes, scores, labels], dim=2)
#             # detections = detections#[:, :100, :]
#             return detections[:, :100, :]
#             # return labels, boxes, scores#, masks

#         # TODO
#         if self.remap_mscoco_category:
#             from ...data.dataset import mscoco_label2category

#             labels = (
#                 torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
#                 .to(boxes.device)
#                 .reshape(labels.shape)
#             )

#         results = []
#         for lab, box, sco in zip(labels, boxes, scores):
#             result = dict(labels=lab, boxes=box, scores=sco)
#             results.append(result)
#         # for lab, box, sco, mask in zip(labels, boxes, scores, masks):
#         #     result = dict(labels=lab, boxes=box, scores=sco, masks=mask)
#         #     results.append(result)

#         return results, targets

#     def deploy(
#         self,
#     ):
#         self.eval()
#         self.deploy_mode = True
#         return self

@register()
class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes, targets=None):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
                
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') # xyxy
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        # TODO for onnx export
        if self.deploy_mode:
            scores = scores.unsqueeze(2)
            labels = labels.unsqueeze(2)
            detections = torch.cat([boxes, scores, labels], dim=2)
            # detections = detections#[:, :100, :]
            return detections[:, :100, :]
            # return labels, boxes, scores#, masks

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results, targets

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self


# class SegmentationPostProcessor(nn.Module):
#     def __call__(self, outputs, targets):
#         # outputs: [N, C, H, W] logits
#         preds = torch.argmax(outputs, dim=1)  # [N, H, W]
#         target_masks = torch.stack([t['mask'] for t in targets])  # [N, H, W]
#         return preds, target_masks

@register()
class SegmentationPostProcessor(nn.Module):
    __share__ = ["do_softmax", "do_argmax"]

    def __init__(
        self, name, do_softmax=True, do_argmax=False
    ) -> None:
        super().__init__()
        self.name = name
        self.do_softmax = do_softmax
        self.do_argmax = do_argmax
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"do_softmax={self.do_softmax}, do_argmax={self.do_argmax}, deploy_mode={self.deploy_mode}"

    def forward(self, outputs, orig_target_sizes: torch.Tensor = None, targets: List[dict]=None):
        """
        outputs: Dict[str, Tensor], expected to contain self.name key.
        Returns a Tensor suitable for MeanIoU: either class indices (N, H, W) or probabilities (N, C, H, W).
        """
        assert self.name in outputs, f"Output does not contain key '{self.name}'"

        preds = outputs[self.name]  # shape: (N, C, H, W)

        if self.do_softmax:
            preds = torch.softmax(preds, dim=1)  # along channel dim C

        if self.do_argmax:
            preds = torch.argmax(preds, dim=1)  # shape: (N, H, W)

        if self.deploy_mode:
            return preds  # Для экспорта в ONNX или inf

        # обработка targets
        if targets is not None:
            # список словарей: [{'mask': Tensor, 'orig_size': Tensor}, ...]
            target_masks = torch.stack([t["masks"] for t in targets], dim=0)  # (N, H, W) или (N, C, H, W) если one-hot

            if self.do_argmax:
                # Если target в виде one-hot или softmax (N, C, H, W), и явно C > 1
                if target_masks.ndim == 4 and target_masks.shape[1] > 1:
                    target_masks = torch.argmax(target_masks, dim=1)  # (N, H, W)
                elif target_masks.ndim == 4 and target_masks.shape[1] == 1:
                    # squeeze лишний канал если C=1
                    target_masks = target_masks.squeeze(1)  # (N, H, W)
            else:
                # если предсказания оставлены в one-hot или probabilities — привести target в такой же формат
                if target_masks.ndim == 3:  # (N, H, W)
                    # например, преобразовать в one-hot
                    num_classes = preds.shape[1]  # C
                    target_masks = torch.nn.functional.one_hot(target_masks.long(), num_classes=num_classes)  # (N, H, W, C)
                    target_masks = target_masks.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        else:
            target_masks = None

        if target_masks is not None and preds.shape != target_masks.shape:
            raise RuntimeError(f"Shape mismatch: preds {preds.shape} vs targets {target_masks.shape}")

        return preds, target_masks  # if MeanIoU configured to accept one-hot / probabilities

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self