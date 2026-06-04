_base_ = ["./tsm_incar_phase1.py"]

model = dict(backbone=dict(frozen_stages=-1))

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=20, norm_type=2),
)

param_scheduler = [
    dict(type="MultiStepLR", begin=0, end=100, by_epoch=True, milestones=[40, 80], gamma=0.1)
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)
work_dir = "./work_dirs/tsm_incar_phase2"
# python mmaction2/tools/train.py configs/tsm_incar_phase2.py --work-dir work_dirs/tsm_incar_phase2 --cfg-options load_from=work_dirs/tsm_incar_phase1/best_acc_top1_epoch_XX.pth --amp