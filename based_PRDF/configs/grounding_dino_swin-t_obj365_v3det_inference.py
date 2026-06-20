_base_ = (
    '../third_party/mmdetection/configs/mm_grounding_dino/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
)

# Activation checkpointing is only useful during training and requires fairscale.
model = dict(
    backbone=dict(with_cp=False),
    encoder=dict(num_cp=0),
    language_model = dict(
        name='./checkpoints/bert-base-uncased'
    )
)
