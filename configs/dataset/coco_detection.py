num_classes=4
remap_mscoco_category=True

transforms_det=dict(
    type='Compose',
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

transforms_eval=dict(
    type='Compose',
    ops=[
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

transforms_semseg=dict(
    type='Compose',
    ops=[
        # dict(type='RandomPhotometricDistort', p=0.15),
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
        ]
    )
)

dataloaders = [
    dict(
        # Метаинформация
        name='train_dataloader',
        role='train',
        # evaluator=dict(
        #     type='DetectionEvaluator',
        #     iou_types=['bbox']
        # ),
        # Аргументы torch.utils.data.DataLoader(**dataloader_params)
        dataloader=dict(
            type='DataLoader',
            dataset=dict(
                type='CocoDetection',
                img_folder='data/COCO2017/train2017/',
                ann_file='data/COCO2017/annotations/instances_train2017.json',
                return_masks=False,
                transforms=transforms_det
            ),
            collate_fn=dict(
                type='BatchImageCollateFunction',
                base_size=576,
                ema_restart_decay=0.9999,
                base_size_repeat=6,
                stop_epoch=48
            ),
            shuffle=True,
            num_workers=4,
            drop_last=True,
            total_batch_size=16,
        ),
    ),
    dict(
        # Метаинформация
        name='val_dataloader',
        role='val',
        evaluator=dict(
            type='DetectionEvaluator',
            iou_types=['bbox'],
            box_format='xyxy', # "xyxy", "xywh", "cxcywh"
            device='cuda'
        ),
        # для постобработки под нужный вывод evaluator
        postprocessor=dict(
            type='DFINEPostProcessor',
            num_classes=num_classes,
            num_top_queries=300
        ),
        # Аргументы torch.utils.data.DataLoader(**dataloader_params)
        dataloader=dict(
            type='DataLoader',
            dataset=dict(
                type='CocoDetection',
                img_folder='data/COCO2017/val2017/',
                ann_file='data/COCO2017/annotations/instances_val2017.json',
                return_masks=False,
                transforms=transforms_eval
            ),
            collate_fn=dict(
                type='BatchImageCollateFunction'
            ),
            shuffle=False,
            num_workers=4,
            total_batch_size=8,
            drop_last=False,
        ),
    ),
    dict(
        name='train_dataloader_tractor_plowed',
        role='train',
        dataloader=dict(
            type='DataLoader',
            dataset=dict(
                type='SemSegmentation',
                img_folder='/data/Datasets/tractor_shared/segm/tractor_plowed/',
                split='train',
                dataset_source="tractor_plowed",
                transforms=transforms_semseg
            ),
            collate_fn=dict(
                type='BatchImageCollateFunction',
                base_size=576,
                ema_restart_decay=0.9999,
                base_size_repeat=6,
                stop_epoch=48
            ),
            shuffle=True,
            num_workers=4,
            drop_last=True,
            total_batch_size=16,
        )
    ),
    dict(
        name='val_dataloader_tractor_plowed',
        role='val',
        evaluator=dict(
            type='SegmentationEvaluator',
            metric='MeanIoU',
            num_classes=6,
            input_format='index',
            per_class=True,
            device='cuda'
        ),
        # для постобработки под нужный вывод
        postprocessor=dict(
            type='SegmentationPostProcessor',
            name='pred_sgm_head_tractor_plowed_main',
            do_softmax=True,
            do_argmax=True
        ),
        dataloader=dict(
            type='DataLoader',
            dataset=dict(
                type='SemSegmentation',
                img_folder='/data/Datasets/tractor_shared/segm/tractor_plowed/',
                split='val',
                dataset_source="tractor_plowed",
                transforms=transforms_eval
            ),
            collate_fn=dict(
                type='BatchImageCollateFunction'
            ),
            shuffle=False,
            num_workers=4,
            total_batch_size=8,
            drop_last=False,
        )
    ),
    dict(
        name='train_dataloader_agro_drivable',
        role='train',
        dataloader=dict(
            type='DataLoader',
            dataset=dict(
                type='SemSegmentation',
                img_folder='/data/Datasets/tractor_shared/segm/agro_drivable/',
                split='train',
                dataset_source="agro_drivable",
                transforms=transforms_semseg
            ),
            collate_fn=dict(
                type='BatchImageCollateFunction',
                base_size=576,
                base_size_repeat=3,
                stop_epoch=72
            ),
            shuffle=True,
            num_workers=4,
            drop_last=True,
            total_batch_size=16
        )
    ),
    dict(
        name='val_dataloader_agro_drivable',
        role='val',
        evaluator=dict(
            type='SegmentationEvaluator',
            metric='MeanIoU',
            num_classes=5,
            input_format='index',
            per_class=True,
            device='cuda'
        ),
        # для постобработки под нужный вывод
        postprocessor=dict(
            type='SegmentationPostProcessor',
            name='pred_sgm_head_agro_drivable_main',
            do_softmax=True,
            do_argmax=True
        ),
        dataloader=dict(
            type='DataLoader',
            dataset=dict(
                type='SemSegmentation',
                img_folder='/data/Datasets/tractor_shared/segm/agro_drivable/',
                split='val',
                dataset_source="agro_drivable",
                transforms=transforms_eval
            ),
            collate_fn=dict(
                type='BatchImageCollateFunction'
            ),
            shuffle=False,
            num_workers=4,
            drop_last=False,
            total_batch_size=8
        )
    )
]