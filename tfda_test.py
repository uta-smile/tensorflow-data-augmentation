#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Dec 16, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : tfda_test.py
project  : tfda
license  : GPL-3.0+

TFDA test
"""

# Tensorflow
import tensorflow as tf

# Others
from tqdm import tqdm

# tf.config.run_functions_eagerly(True)
# tf.debugging.set_log_device_placement(True)
tf.config.set_visible_devices([], "GPU")

# Local
from tfda.augmentations.utils import to_one_hot
from tfda.base import Compose
from tfda.defs import TFDAData, TFDADefault3DParams, nan, pi
from tfda.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from tfda.transforms.custom_transforms import MaskTransform, OneHotTransform, OneHotTransform2D
from tfda.transforms.noise_transforms import (
    GaussianBlurTransform, GaussianBlurTransform2D,
    GaussianNoiseTransform,
)
from tfda.transforms.resample_transforms import SimulateLowResolutionTransform, SimulateLowResolutionTransform2D
from tfda.transforms.spatial_transforms import (
    MirrorTransform, MirrorTransform2D,
    SpatialTransform,
)
from tfda.transforms.utility_transforms import RemoveLabelTransform

params = TFDADefault3DParams(
    rotation_x=(
        -30.0 / 360 * 2.0 * pi,
        30.0 / 360 * 2.0 * pi,
    ),
    rotation_y=(
        -30.0 / 360 * 2.0 * pi,
        30.0 / 360 * 2.0 * pi,
    ),
    rotation_z=(
        -30.0 / 360 * 2.0 * pi,
        30.0 / 360 * 2.0 * pi,
    ),
    scale_range=(0.7, 1.4),
    do_elastic=False,
    selected_seg_channels=[0],
    patch_size_for_spatial_transform=[40, 56, 40],
    num_cached_per_thread=2,
    mask_was_used_for_normalization=nan,
)


def all_da():
    da = Compose(
        [
            tf.keras.layers.Input(
                type_spec=TFDAData.Spec(
                    None, tf.TensorSpec(None), tf.TensorSpec(None), tf.TensorSpec(None), tf.TensorSpec(None)
                )
            ),
            # SpatialTransform(
            #     patch_size=params.patch_size_for_spatial_transform,
            #     patch_center_dist_from_border=nan,
            #     do_elastic_deform=params.do_elastic,
            #     alpha=params.elastic_deform_alpha,
            #     sigma=params.elastic_deform_sigma,
            #     do_rotation=params.do_rotation,
            #     angle_x=params.rotation_x,
            #     angle_y=params.rotation_y,
            #     angle_z=params.rotation_z,
            #     p_rot_per_axis=params.rotation_p_per_axis,
            #     do_scale=params.do_scaling,
            #     scale=params.scale_range,
            #     border_mode_data=params.border_mode_data,
            #     border_cval_data=0.0,
            #     order_data=3.0,
            #     border_mode_seg="constant",
            #     border_cval_seg=-1.0,
            #     order_seg=1.0,
            #     random_crop=params.random_crop,
            #     p_el_per_sample=params.p_eldef,
            #     p_scale_per_sample=params.p_scale,
            #     p_rot_per_sample=params.p_rot,
            #     independent_scale_for_each_axis=params.independent_scale_factor_for_each_axis,
            # ),
            GaussianNoiseTransform(p_per_channel=0.01),
            GaussianBlurTransform2D(
                (0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            ),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.15
            ),
            ContrastAugmentationTransform(p_per_sample=0.15),
            SimulateLowResolutionTransform2D(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
            ),
            GammaTransform(
                (0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1
            ),
            GammaTransform(
                (0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3
            ),
            MirrorTransform2D((0, 1, 2)),
            MaskTransform(
                tf.constant([[0, 0]]), mask_idx_in_seg=0, set_outside_to=0.0
            ),
            RemoveLabelTransform(-1, 0),
            OneHotTransform2D([0, 1]),
        ]
    )
    da.compile()
    da.summary()

    dataseti = iter(
        tf.data.Dataset.from_tensor_slices(
            tf.random.uniform((2 * 8 * 1 * 73 * 80 * 8 * 8,), 0, 100)
        )
        .batch(64)
        .batch(80)
        .batch(1)
        .batch(8)
        .map(lambda x: da(TFDAData(x, x)))
        .prefetch(tf.data.AUTOTUNE)
    )
    res = []
    for dataset in tqdm(dataseti, desc="steps:"):

        res.append(dataset)

    # assert len(res) == 100
    for d in res:
        r = d.data
        tf.print(r.shape)
        # assert r.shape[0] == 2
        # assert r.shape[1] == 40
        # assert r.shape[2] == 56
        # assert r.shape[3] == 40
        assert r.shape[0] == 8
        assert r.shape[1] == 73
        assert r.shape[2] == 80
        assert r.shape[3] == 64
        assert r.shape[4] == 1
    return res


def test():
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    with tf.device("/CPU:0"):
        res = all_da()
    # import pdb;pdb.set_trace()


if __name__ == "__main__":
    test()
