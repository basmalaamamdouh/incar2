_base_ = [
    "../mmaction2/configs/recognition/tsm/"
    "tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb.py"
]
custom_imports = dict(imports=['mmaction.models', 'mmaction.datasets', 'mmaction.evaluation'], allow_failed_imports=False)

model = dict(
    backbone=dict(frozen_stages=3, num_segments=16),
    cls_head=dict(
        num_classes=2,
        in_channels=2048,
        dropout_ratio=0.5,
        init_std=0.001,
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.83, 1.0]  # VIOLENT=1.83, NONVIOLENT=1.0
            # class_weight=[1.0, 1.83]  # VIOLENT=1.0, NONVIOLENT=1.83
        ),
        num_segments=16,
    ),
)

INCAR_ROOT = r"D:/incar/dataset/INCAR2c_thermal_violent/rawframes"
data_root = 'D:/incar/dataset/INCAR2c_thermal_violent/rawframes'
data_root_val = 'D:/incar/dataset/INCAR2c_thermal_violent/rawframes'
dataset_type = "RawframeDataset"
# dataset_type = "VideoDataset"

train_pipeline = [
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=16),
    dict(type="RawFrameDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    # dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

val_pipeline = [
    # dict(type="UniformSample", clip_len=1, num_clips=1, test_mode=True),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=16, test_mode=True),
    dict(type="RawFrameDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    # dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,  # disabled - caused CUDA invalid argument
    sampler=dict(type='DefaultSampler', shuffle=True),    dataset=dict(
        type=dataset_type,
        ann_file=f"{INCAR_ROOT}/recheck_INCAR2c_train_rawframes.txt",
        data_prefix=dict(img=INCAR_ROOT),
        filename_tmpl="img_{:05d}.jpg",
        pipeline=train_pipeline,
        num_classes=2,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=f"{INCAR_ROOT}/recheck_INCAR2c_val_rawframes.txt",
        data_prefix=dict(img=INCAR_ROOT),
        filename_tmpl="img_{:05d}.jpg",
        pipeline=val_pipeline,
        num_classes=2,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=0,   # 0 to avoid worker process issues
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=f"{INCAR_ROOT}/recheck_INCAR2c_test_rawframes.txt",
        data_prefix=dict(img=INCAR_ROOT),
        filename_tmpl="img_{:05d}.jpg",
        pipeline=test_pipeline,
        num_classes=2,
        test_mode=True,
    ),
)

val_evaluator = dict(type="AccMetric")
test_evaluator = dict(type="AccMetric")

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=20, norm_type=2),
)

param_scheduler = [
    dict(type="MultiStepLR", begin=0, end=50, by_epoch=True, milestones=[20, 40], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="acc/top1", rule="greater"),
    logger=dict(type="LoggerHook", interval=20),
)

work_dir = "./work_dirs/tsm_incar_phase1_v3"
# python -s mmaction2/tools/test.py D:\incar\incar2c_thermal_pipeline\configs\tsm_incar_phase1.py D:\incar\incar2c_thermal_pipeline\work_dirs\tsm_incar_phase1\best_acc_top1_epoch_45.pth --dump D:\incar\incar2c_thermal_pipeline\results\test_predictions_phase1.pkl
# python -s mmaction2/tools/test.py D:\incar\incar2c_thermal_pipeline\configs\tsm_incar_phase2.py D:\incar\incar2c_thermal_pipeline\work_dirs\tsm_incar_phase2\best_acc_top1_epoch_64.pth --dump D:\incar\incar2c_thermal_pipeline\results\test_predictions_phase2.pkl
