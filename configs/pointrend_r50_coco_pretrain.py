_base_ = ['../mmdetection/configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py']

model = dict(roi_head=dict(
    bbox_head=dict(num_classes=1),
    mask_head=dict(num_classes=1),
    point_head=dict(num_classes=1),
))

dataset_type = 'CocoDataset'
data_root = 'data/airs_proto/'
classes = 'rooftop',

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(512, 512)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_annotations.json',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_annotations.json',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_annotations.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        classes=classes,
    ),
)
evaluation = dict(metric=['bbox', 'segm'])

optimizer = dict(type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    gamma=0.5, # reduce the lr twice at each epoch in `step`
    step=[8, 11],
)
runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(hooks=[
    dict(type='TextLoggerHook', interval=10),
    dict(type='TensorboardLoggerHook', interval=1, by_epoch=True)
])

load_from = './pretrained_models/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth'
work_dir = './training_logs/point_rend_r50'
