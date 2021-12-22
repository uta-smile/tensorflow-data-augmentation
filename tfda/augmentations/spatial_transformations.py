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
filename : spatial_transformations.py
project  : augmentations
license  : GPL-3.0+

spatial transformations
"""

import tensorflow as tf
from tfda.base import TFT
from tfda.utils import (
    TFbF,
    TFbT,
    TFf0,
    TFf1,
    TFi1,
    to_tf_bool,
    to_tf_float,
    pi,
)



def augment_spatial_helper(sample_id:TFT, patch_size: TFT) -> TFT:
    pass




def augment_spatial(
    data: TFT,
    seg: TFT,
    patch_size: TFT,
    patch_center_dist_from_border: TFT = 30 * TFi1,
    do_elastic_deform: TFT = TFbT,
    alpha: TFT = None,
    sigma: TFT = None,
    do_rotation: TFT = TFbT,
    angle_x: TFT = None,
    angle_y: TFT = None,
    angle_z: TFT = None,
    do_scale: TFT = TFbT,
    scale: TFT = None,
    border_mode_data: TFT = None,
    border_cval_data: TFT = TFf0,
    order_data: TFT = 3 * TFf1,
    border_mode_seg: TFT = None,
    border_cval_seg: TFT = TFf0,
    order_seg: TFT = TFf0,
    random_crop: TFT = TFbT,
    p_el_per_sample: TFT = TFf1,
    p_scale_per_sample: TFT = TFf1,
    p_rot_per_sample: TFT = TFf1,
    independent_scale_for_each_axis: TFT = TFbF,
    p_rot_per_axis: TFT = TFf1,
    p_independent_scale_per_axis: TFT = TFf1,
) -> TFT:
    # init
    alpha = tf.cond(
        to_tf_bool(alpha is None), lambda: to_tf_float((0.0, 1000.0)), alpha
    )
    sigma = tf.cond(
        to_tf_bool(sigma is None), lambda: to_tf_float((10.0, 13.0)), sigma
    )
    angle_x = tf.cond(
        to_tf_bool(angle_x is None), lambda: to_tf_float((0, 2 * pi)), angle_x
    )
    angle_y = tf.cond(
        to_tf_bool(angle_y is None), lambda: to_tf_float((0, 2 * pi)), angle_y
    )
    angle_z = tf.cond(
        to_tf_bool(angle_z is None), lambda: to_tf_float((0, 2 * pi)), angle_z
    )
    scale = tf.cond(
        to_tf_bool(scale is None),
        lambda: to_tf_float((0.75, 1.25)),
        lambda: scale,
    )
    border_mode_data = tf.cond(
        to_tf_bool(border_mode_data is None),
        lambda: tf.cast("nearest", tf.string),
        lambda: border_mode_data,
    )
    border_mode_seg = tf.cond(
        to_tf_bool(border_mode_seg is None),
        lambda: tf.cast("constant", tf.string),
        lambda: border_mode_seg,
    )

    # start here
    dim = patch_size.shape[0]

    seg_result = tf.cond(
        to_tf_bool(seg is None),
        lambda: None,
        lambda: tf.cond(
            dim == 2,
            lambda: tf.zeros(
                (seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]),
                tf.float32,
            ),
            lambda: tf.zeros(
                (
                    seg.shape[0],
                    seg.shape[1],
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                ),
                tf.float32,
            ),
        ),
    )

    data_result = tf.cond(
        dim == 2,
        lambda: tf.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1]),
            tf.float32,
        ),
        lambda: tf.zeros(
            (
                data.shape[0],
                data.shape[1],
                patch_size[0],
                patch_size[1],
                patch_size[2],
            ),
            tf.float32,
        ),
    )

    # Never used
    # if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
    #     patch_center_dist_from_border = dim * [patch_center_dist_from_border]
