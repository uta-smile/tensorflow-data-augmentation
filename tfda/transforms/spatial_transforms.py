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
date     : Dec 23, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : spatial_transforms.py
project  : transforms
license  : GPL-3.0+

Spatial Transforms
"""

from typing import Tuple
import tensorflow as tf

# Others
from tfda.augmentations.spatial_transformations import (
    augment_mirroring,
    augment_spatial,
)
from tfda.base import DTFT, TFT, TFDABase
from tfda.utils import TFbF, TFbT, TFf0, TFf1, TFi1, pi, nan


class SpatialTransform(TFDABase):
    def __init__(
        self,
        patch_size: TFT,
        patch_center_dist_from_border: TFT = 30 * TFi1,
        do_elastic_deform: TFT = TFbT,
        alpha: TFT = (0.0, 1000.0),
        sigma: TFT = (10.0, 13.0),
        do_rotation: TFT = TFbT,
        angle_x: TFT = (0, 2 * pi),
        angle_y: TFT = (0, 2 * pi),
        angle_z: TFT = (0, 2 * pi),
        do_scale: TFT = TFbT,
        scale: TFT = (0.75, 1.25),
        border_mode_data: TFT = "nearest",
        border_cval_data: TFT = TFf0,
        order_data: TFT = 3 * TFf1,
        border_mode_seg: TFT = "constant",
        border_cval_seg: TFT = TFf0,
        order_seg: TFT = TFf0,
        random_crop: TFT = TFbT,
        data_key: TFT = "data",
        label_key: TFT = "seg",
        p_el_per_sample: TFT = TFf1,
        p_scale_per_sample: TFT = TFf1,
        p_rot_per_sample: TFT = TFf1,
        independent_scale_for_each_axis: TFT = TFbF,
        p_rot_per_axis: TFT = TFf1,
        p_independent_scale_per_axis: TFT = TFf1,
        **kws,
    ) -> None:
        super().__init__(**kws)
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.data_key = data_key
        self.label_key = label_key
        self.p_el_per_sample = p_el_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_rot_per_sample = p_rot_per_sample
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def call(self, data_dict: DTFT) -> DTFT:
        data = data_dict.get(self.data_key, nan)
        seg = data_dict.get(self.label_key, nan)

        # if self.patch_size is None:
        #     if len(data.shape) == 4:
        #         patch_size = (data.shape[2], data.shape[3])
        #     elif len(data.shape) == 5:
        #         patch_size = (data.shape[2], data.shape[3], data.shape[4])
        #     else:
        #         raise ValueError("only support 2D/3D batch data.")
        # else:
        #     patch_size = self.patch_size

        patch_size = self.patch_size

        ret_val = augment_spatial(
            data,
            seg,
            patch_size=patch_size,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            do_elastic_deform=self.do_elastic_deform,
            alpha=self.alpha,
            sigma=self.sigma,
            do_rotation=self.do_rotation,
            angle_x=self.angle_x,
            angle_y=self.angle_y,
            angle_z=self.angle_z,
            do_scale=self.do_scale,
            scale=self.scale,
            border_mode_data=self.border_mode_data,
            border_cval_data=self.border_cval_data,
            order_data=self.order_data,
            border_mode_seg=self.border_mode_seg,
            border_cval_seg=self.border_cval_seg,
            order_seg=self.order_seg,
            random_crop=self.random_crop,
            p_el_per_sample=self.p_el_per_sample,
            p_scale_per_sample=self.p_scale_per_sample,
            p_rot_per_sample=self.p_rot_per_sample,
            independent_scale_for_each_axis=self.independent_scale_for_each_axis,
            p_rot_per_axis=self.p_rot_per_axis,
            p_independent_scale_per_axis=self.p_independent_scale_per_axis,
        )

        data_dict[self.data_key] = ret_val[0]
        if not tf.math.reduce_any(tf.math.is_nan(seg)):
            data_dict[self.label_key] = ret_val[1]
        return data_dict


class MirrorTransform(TFDABase):
    """Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), **kws):
        super().__init__(**kws)
        self.axes = axes

    def call(self, data_dict: DTFT) -> DTFT:
        data = data_dict.get(self.data_key, nan)
        seg = data_dict.get(self.label_key, nan)

        data_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        seg_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for b in tf.range(tf.shape(data)[0]):
            if tf.random.uniform(()) < self.p_per_sample:
                sample_seg = nan
                if not tf.math.reduce_any(tf.math.is_nan(seg)):
                    sample_seg = seg[b]
                ret_val = augment_mirroring(
                    data[b], sample_seg, axes=self.axes
                )
                data_list = data_list.write(b, ret_val[0])
                if not tf.math.reduce_any(tf.math.is_nan(seg)):
                    seg_list = seg_list.write(b, ret_val[1])

        data_dict["data"] = data_list.stack()
        if tf.rank(seg) > 0 and seg_list.size() > 0:
            data_dict["seg"] = seg_list.stack()

        return data_dict


if __name__ == "__main__":
    # dataset = (
    #     tf.data.Dataset.range(8 * 1 * 1 * 70 * 83 * 64, output_type=tf.float32)
    #     .batch(64)
    #     .batch(83)
    #     .batch(70)
    #     .batch(1)
    #     .batch(1)
    # )

    # di = iter(dataset)
    # data_sample = next(di)
    # seg_sample = next(di)

    data_sample = tf.random.uniform((1, 1, 70, 83, 64))
    seg_sample = tf.random.uniform((1, 1, 70, 83, 64))

    sa = SpatialTransform(tf.cast([40, 56, 40], tf.int64), random_crop=TFbF)

    with tf.device("/CPU:0"):
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        # tf.print(sa(data=data_sample, seg=seg_sample))

        images = tf.random.uniform((8, 2, 20, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.float32
        )
        data_dict = {"data": images, "seg": labels}
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
        data_dict = MirrorTransform((0, 1, 2))(dict(data=images, seg=labels))
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)

    tf.print("END")
