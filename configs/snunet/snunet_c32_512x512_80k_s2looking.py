_base_ = 'snunet_c16_512x512_80k_s2loooking.py'

base_channels = 32
model = dict(
    backbone=dict(base_channel=base_channels),
    decode_head=dict(
        in_channels=base_channels * 4,
        channels=base_channels * 4))
