_base_ = [
    '../../dataset/coco_detection.py',
    '../../runtime.py',
    # '../include/dataloader.py', # dublicate, was moved to coco_detection.py
    '../include/optimizer.py',
    '../include/dfine_hgnetv2.py',
]

output_dir = './output/dfine_hgnetv2_m_obj2coco'

use_ema = True  # Enabled by default
use_amp = True

model='DFINE'

DFINE=dict(
  backbone='HGNetv2',
  encoder='HybridEncoder',
  decoder='DFINETransformer',
  spatial='SpatialPath',
  sgm_decoder='BiSeNetDecoder'
)

# HGNetv2 settings
HGNetv2 = dict(
    name='B2',
    return_idx=[1, 2, 3],
    freeze_at=-1,
    freeze_norm=False,
    use_lab=True,
    pretrained=True,
    local_model_dir='weight/hgnetv2/'
)

BiSeNetDecoder=dict(
  in_planes=[768, 1536],
  head_configs=[
      dict(
          name='tractor_plowed',
          out_planes=6
      ),
      dict(
          name='agro_drivable',
          out_planes=5
      )
  ]
)

DFINETransformer=dict(
  feat_channels=[256, 256, 256],
  feat_strides=[8, 16, 32],
  hidden_dim=256,
  num_levels=3,

  num_layers=4,
  eval_idx=-1,
  num_queries=300,

  num_denoising=100,#100 #0 #100
  label_noise_ratio=0.5,
  box_noise_scale=1.0,

  # NEW
  reg_max=32,
  reg_scale=4,

  # Auxiliary decoder layers dimension scaling
  # "eg. If num_layers=6 eval_idx=-4,
  # then layer 3, 4, 5 are auxiliary decoder layers."
  layer_scale=1,  # 2


  num_points=[3, 6, 3], # [4, 4, 4] [3, 6, 3]
  cross_attn_method='default', # default, discrete
  query_select_method='default', # default, agnostic
  freeze=False # False by default !!!
)

HybridEncoder=dict(
  in_channels=[384, 768, 1536],
  feat_strides=[8, 16, 32],
  # intra
  hidden_dim=256,
  use_encoder_idx=[2],
  num_encoder_layers=1,
  nhead=8,
  dim_feedforward=1024,
  dropout=0.,
  enc_act='gelu',

  # cross
  expansion=1.0,
  depth_mult=0.67,
  act='silu',
  freeze=False
)

SpatialPath=dict(
  in_planes=3,
  out_planes=256 #256 #192 #128 for base BiSeNet, 256 for wider backbone ???
)

optimizer = dict(
    type='AdamW',
    params=[
        dict(
            params='^(?=.*backbone)(?!.*norm|bn).*$',
            lr=0.000025
        ),
        dict(
            params='^(?=.*backbone)(?=.*norm|bn).*$',
            lr=0.000025,
            weight_decay=0.0
        ),
        dict(
            params='^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$',
            weight_decay=0.0
        )
    ],
    lr=0.00025,
    betas=(0.9, 0.999),
    weight_decay=0.000125
)

epochs = 56  # early stop

ema=dict(
  type='ModelEMA',
  decay=0.9999,
  warmups=0,
  start=0
)

lr_warmup_scheduler = dict(
    warmup_duration=0
)

best_criterion = dict(
    dataloader = 'val_dataloader',
    evaluator = 'DetectionEvaluator',
    metric_index = 'map', # например, AP50:95
    mode = 'max'
)
    # # dataloader: 'val_loader_name'
    # # evaluator: 'coco_eval_bbox'  # или 'mean_iou_eval', etc.
    # # metric_index: 0  # например, AP50:95
    # # mode: 'max'  # или 'min' (например, для loss)