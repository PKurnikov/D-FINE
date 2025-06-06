"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'spatial', 'sgm_decoder'] #  , 'spatial', 'sgm_decoder',

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        spatial: nn.Module,
        sgm_decoder: nn.Module
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        # for shared
        self.spatial = spatial
        self.sgm_decoder = sgm_decoder

    def is_det_targets(self, targets=None):
        if targets is None:
            return False
        return any('boxes' in t for t in targets)

    def forward(self, x, targets=None, det_mode=2):
        # x = self.backbone(x)
        # x = self.encoder(x)
        # x = self.decoder(x, targets)

        # return x

        if (targets is None):
            indices_with_masks = list(range(x.shape[0]))
            indices_with_bbox = list(range(x.shape[0]))
            filtered_targets = None
        else:
            # Найти индексы элементов, где есть 'masks'
            indices_with_masks = [i for i, t in enumerate(targets) if 'masks' in t]
            # Найти индексы элементов, где есть 'bbox'
            indices_with_bbox = [i for i, t in enumerate(targets) if 'boxes' in t]
            filtered_targets = [targets[i] for i in indices_with_bbox]
       
        # Создать новый тензор x по выбранным индексам
        x_with_masks = x[indices_with_masks]    
      
        if det_mode > 0:  
            # for shared
            x_spatial = self.spatial(x_with_masks)
        
        x = self.backbone(x)
        
        if det_mode > 0:
            # Создать новый тензор x по выбранным индексам
            x_backbon_segm = [v_[indices_with_masks] for v_ in x]
            x_sgm = self.sgm_decoder(x_spatial, x_backbon_segm)
        
        # x.reverse()
        
        # Создать новый тензор x по выбранным индексам
        filtered_x = [v_[indices_with_bbox] for v_ in x]
        
        filtered_x = self.encoder(filtered_x)
        
        # Создать новый список targets, содержащий только элементы с 'bbox'
        # filtered_targets = [targets[i] for i in indices_with_bbox]
        filtered_x = self.decoder(filtered_x, filtered_targets)
        
        # return filtered_x
        
        if det_mode == 1:
            return x_sgm
        if det_mode == 2:
            filtered_x.update(x_sgm)
        
        return filtered_x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
