task='multitask'
# task=detection

use_focal_loss=True
eval_spatial_size=[576, 1024] # h w

criterion='DFINECriterion'

# центральный постпроцессинг для SHARED вывода и деплоя
postprocessor = dict(
    type='SharedPostProcessor',
    deploy_mode=False,  # включить для ONNX/TorchTrace/TensorRT

    heads=dict(
        segmentation=dict(
            enable=True,
            outputs=['pred_sgm_head_tractor_plowed_main', 
                     'pred_sgm_head_agro_drivable_main'],
            do_softmax=True,
            do_argmax=False,
            do_concat=True      # объединить выходы сегментации по оси C формата NCHW
        ),
        detector=dict(
            enable=True,
            logits_key='pred_logits',
            boxes_key='pred_boxes',
            num_classes=4,
            use_focal_loss=True,
            num_top_queries=100,
            remap_mscoco_category=False,
            in_fmt='cxcywh',
            out_fmt='xyxy',
            scale_factor=4.0
        )
    )
)

ProbOhemCriterion=dict(
    ignore_label=255,
    thresh=0.7,
    min_kept=256,           # расчитывается исходя из формулы: int(batch_size // len(gpus) * inp_height * inp_width // 16)
    down_ratio=1,
    weight=None,
)

DFINECriterion=dict(
    weight_dict=dict(
      loss_vfl=1, 
      loss_bbox=5,
      loss_giou=2, 
      loss_fgl=0.15,
      loss_ddf=1.5,
      pred_sgm_head_tractor_plowed_aux1=1.0,
      pred_sgm_head_tractor_plowed_aux2=1.0,
      pred_sgm_head_tractor_plowed_main=1.0,
      pred_sgm_head_agro_drivable_aux1=1.0,
      pred_sgm_head_agro_drivable_aux2=1.0,
      pred_sgm_head_agro_drivable_main=1.0),

    losses=[],
    # losses=['vfl', 'boxes', 'local'],
    alpha=0.75,
    gamma=2.0,
    reg_max=32,

    matcher=dict(
        type='HungarianMatcher',
        weight_dict=dict(cost_class=2, cost_bbox=5, cost_giou=2),
        alpha=0.25,
        gamma=2.0,
    )
)