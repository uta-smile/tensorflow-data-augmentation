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

import tensorflow as tf

# Types
from typing import Tuple

# Local
# tf.debugging.set_log_device_placement(True)
from tfda.augmentations.spatial_transformations import (
    augment_mirroring,
    augment_spatial
)
from tfda.base import TFDABase
from tfda.defs import TFDAData, TFbF, TFbT, nan, pi
from tfda.utils import isnotnan


class SpatialTransform(TFDABase):
    def __init__(
        self,
        patch_size: tf.Tensor,
        patch_center_dist_from_border: tf.Tensor = 30,
        do_elastic_deform: tf.Tensor = TFbT,
        alpha: tf.Tensor = (0.0, 1000.0),
        sigma: tf.Tensor = (10.0, 13.0),
        do_rotation: tf.Tensor = TFbT,
        angle_x: tf.Tensor = (0, 2 * pi),
        angle_y: tf.Tensor = (0, 2 * pi),
        angle_z: tf.Tensor = (0, 2 * pi),
        do_scale: tf.Tensor = TFbT,
        scale: tf.Tensor = (0.75, 1.25),
        border_mode_data: str = "nearest",
        border_cval_data: tf.Tensor = 0.0,
        order_data: tf.Tensor = 3,
        border_mode_seg: str = "constant",
        border_cval_seg: tf.Tensor = 0.0,
        order_seg: tf.Tensor = 0.0,
        random_crop: tf.Tensor = TFbT,
        data_key: str = "data",
        label_key: str = "seg",
        p_el_per_sample: tf.Tensor = 1.0,
        p_scale_per_sample: tf.Tensor = 1.0,
        p_rot_per_sample: tf.Tensor = 1.0,
        independent_scale_for_each_axis: tf.Tensor = TFbF,
        p_rot_per_axis: tf.Tensor = 1.0,
        p_independent_scale_per_axis: tf.Tensor = 1.0,
        **kws,
    ) -> None:
        super().__init__(**kws)
        self.patch_size = tf.convert_to_tensor(patch_size)
        self.patch_center_dist_from_border = tf.convert_to_tensor(patch_center_dist_from_border)
        self.do_elastic_deform = tf.convert_to_tensor(do_elastic_deform)
        self.alpha = tf.convert_to_tensor(alpha)
        self.sigma = tf.convert_to_tensor(sigma)
        self.do_rotation = tf.convert_to_tensor(do_rotation)
        self.angle_x = tf.convert_to_tensor(angle_x)
        self.angle_y = tf.convert_to_tensor(angle_y)
        self.angle_z = tf.convert_to_tensor(angle_z)
        self.do_scale = tf.convert_to_tensor(do_scale)
        self.scale = tf.convert_to_tensor(scale)
        self.border_mode_data = border_mode_data
        self.border_cval_data = tf.convert_to_tensor(border_cval_data)
        self.order_data = tf.convert_to_tensor(order_data)
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = tf.convert_to_tensor(border_cval_seg)
        self.order_seg = tf.convert_to_tensor(order_seg)
        self.random_crop = tf.convert_to_tensor(random_crop)
        self.data_key = data_key
        self.label_key = label_key
        self.p_el_per_sample = tf.convert_to_tensor(p_el_per_sample)
        self.p_scale_per_sample = tf.convert_to_tensor(p_scale_per_sample)
        self.p_rot_per_sample = tf.convert_to_tensor(p_rot_per_sample)
        self.independent_scale_for_each_axis = tf.convert_to_tensor(independent_scale_for_each_axis)
        self.p_rot_per_axis = tf.convert_to_tensor(p_rot_per_axis)
        self.p_independent_scale_per_axis = tf.convert_to_tensor(p_independent_scale_per_axis)

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        data = dataset.data
        seg = dataset.seg

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

        data = ret_val[0]
        if isnotnan(seg)):
            seg = ret_val[1]
        return TFDAData(data, seg)


class MirrorTransform(TFDABase):
    """Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes: tf.Tensor = (0, 1, 2), **kws):
        super().__init__(**kws)
        axes = tf.cast(axes, tf.int64)
        self.axes = axes

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        data = dataset.data
        seg = dataset.seg

        data_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        seg_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for b in tf.range(tf.shape(data)[0]):
            if tf.random.uniform(()) < self.defs.p_per_sample:
                sample_seg = nan
                if isnotnan(seg)):
                    sample_seg = seg[b]
                ret_val = augment_mirroring(
                    data[b], sample_seg, axes=self.axes
                )
                data_list = data_list.write(b, ret_val[0])
                if isnotnan(seg)):
                    seg_list = seg_list.write(b, ret_val[1])

        data = data_list.stack()
        if tf.rank(seg) > 0 and seg_list.size() > 0:
            seg = seg_list.stack()

        return TFDAData(data, seg)


@tf.function
def test():
    images = tf.random.uniform((8, 2, 20, 376, 376))
    labels = tf.random.uniform(
        (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.float32
    )
    data_dict = TFDAData(images, labels)
    # tf.print(
    #     data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
    # )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
    data_dict = MirrorTransform((0, 1, 2))(TFDAData(data=images, seg=labels))
    # tf.print(
    #     data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
    # )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
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

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    with tf.device("/CPU:0"):
        tf.print(sa(TFDAData(data=data_sample, seg=seg_sample)))
        tf.print(test()["data"].shape)

        tf.print("END")
