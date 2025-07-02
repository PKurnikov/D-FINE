# config.py

dataloaders = [
    dict(
        name='train_dataloader',
        role='train',
        dataset=dict(
            transforms=dict(
                ops=[
                    dict(type='RandomPhotometricDistort', p=0.5),
                    dict(type='RandomZoomOut', fill=0),
                    dict(type='RandomIoUCrop', p=0.8),
                    dict(type='SanitizeBoundingBoxes', min_size=1),
                    dict(type='RandomHorizontalFlip'),
                    dict(type='Resize', size=[576, 1024]),
                    dict(type='SanitizeBoundingBoxes', min_size=1),
                    dict(type='ConvertPILImage', dtype='float32', scale=True),
                    dict(type='ConvertBoxes', fmt='cxcywh', normalize=True),
                ],
                policy=dict(
                    name='stop_epoch',
                    epoch=72,
                    ops=[
                        'RandomPhotometricDistort',
                        'RandomZoomOut',
                        'RandomIoUCrop',
                    ]
                )
            )
        ),
        collate_fn=dict(
            type='BatchImageCollateFunction',
            base_size=576,
            base_size_repeat=3,
            stop_epoch=72
        ),
        shuffle=True,
        total_batch_size=4,
        num_workers=4
    ),
    dict(
        name='train_dataloader_tractor_plowed',
        role='train',
        dataset=dict(
            transforms=dict(
                ops=[
                    dict(type='RandomPhotometricDistort', p=0.15),
                    dict(type='RandomHorizontalFlip'),
                    dict(type='RandomCrop', p=0.25, size=[576, 576]),
                    dict(type='Resize', size=[576, 1024]),
                    dict(type='ConvertPILImage', dtype='float32', scale=True),
                ],
                policy=dict(
                    name='stop_epoch',
                    epoch=72,
                    ops=[
                        'RandomPhotometricDistort',
                        'RandomZoomOut',
                        'RandomIoUCrop',
                    ]
                )
            )
        ),
        collate_fn=dict(
            type='BatchImageCollateFunction',
            base_size=576,
            base_size_repeat=3,
            stop_epoch=72
        ),
        shuffle=True,
        total_batch_size=4,
        num_workers=4
    ),
]