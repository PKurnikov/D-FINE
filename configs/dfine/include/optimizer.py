use_amp=False
use_ema=False
ema=dict(
  type='ModelEMA',
  decay=0.9999,
  warmups=1000,
  start=0
)

epochs=72
clip_max_norm=0.1


optimizer=dict(
    type='AdamW',
    params=[
        dict(
            params='^(?=.*backbone)(?!.*norm).*$',
            lr=0.0000125
       ),
        dict(
            params='^(?=.*backbone)(?=.*norm|bn).*$',
            lr=0.0000125,
            weight_decay=0.
       ),
        dict(
            params='^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$',
            weight_decay=0.
        )
    ],
    lr=0.000125,#0.00025,
    betas=[0.9, 0.999],
    weight_decay=0.000125
)

lr_scheduler=dict(
  type='MultiStepLR',
  milestones=[500],
  gamma=0.1,
)

lr_warmup_scheduler=dict(
  type='LinearWarmup',
  warmup_duration=500
)
