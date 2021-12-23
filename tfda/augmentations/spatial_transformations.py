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

# Types
from typing import Optional

tf.debugging.set_log_device_placement(True)
# Others
from tfda.augmentations.utils import (
    create_zero_centered_coordinate_mesh,
    elastic_deform_coordinates,
    rotate_coords_2d,
    rotate_coords_3d,
    scale_coords,
)
from tfda.base import TFT
from tfda.utils import (
    TFbF,
    TFbT,
    TFf0,
    TFf1,
    TFi1,
    pi,
    to_tf_bool,
    to_tf_float,
)


def augment_spatial_helper(sample_id: TFT, patch_size: TFT) -> TFT:
    pass


@tf.function(experimental_follow_type_hints=True)
def augment_spatial(
    data: TFT,
    seg: Optional[TFT],
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
    p_el_per_sample: TFT = TFf1,
    p_scale_per_sample: TFT = TFf1,
    p_rot_per_sample: TFT = TFf1,
    independent_scale_for_each_axis: TFT = TFbF,
    p_rot_per_axis: TFT = TFf1,
    p_independent_scale_per_axis: TFT = TFf1,
) -> TFT:
    # start here
    dim = tf.shape(patch_size)[0]

    # don't do it now!
    # seg_result, data_result

    # Never used
    # if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
    #     patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in tf.range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        # cshape = tf.concat([[coords.get_shape()[0]], patch_size], 0)
        modified_coords = TFbF
        if do_elastic_deform and tf.less(
            tf.random.uniform(()), p_el_per_sample
        ):
            a = tf.random.uniform((), alpha[0], alpha[1])
            s = tf.random.uniform((), sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = TFbT

        if do_rotation and tf.less(tf.random.uniform(()), p_rot_per_sample):
            if tf.less_equal(tf.random.uniform(()), p_rot_per_axis):
                a_x = tf.random.uniform((), angle_x[0], angle_x[1])
            else:
                a_x = TFf0

            if to_tf_bool(tf.equal(dim, 3)):
                if tf.less_equal(tf.random.uniform(()), p_rot_per_axis):
                    a_y = tf.random.uniform((), angle_y[0], angle_y[1])
                else:
                    a_y = TFf0

                if tf.less_equal(tf.random.uniform(()), p_rot_per_axis):
                    a_z = tf.random.uniform((), angle_z[0], angle_z[1])
                else:
                    a_z = TFf0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                pass
                # coords = rotate_coords_2d(coords, a_x)
            modified_coords = TFbT

        if do_scale and tf.less(tf.random.uniform(()), p_scale_per_sample):
            if independent_scale_for_each_axis and tf.less(
                tf.random.uniform(()), p_independent_scale_per_axis
            ):
                sc = tf.map_fn(
                    lambda x: tf.cond(
                        tf.less(tf.random.uniform(()), tf.constant(0.5))
                        and tf.less(scale[0], tf.constant(1.0)),
                        lambda: tf.random.uniform((), scale[0], 1.0),
                        lambda: tf.random.uniform(
                            (), tf.maximum(scale[0], 1.0), scale[1]
                        ),
                    ),
                    tf.range(dim, dtype=tf.float32),
                )
            else:
                sc = tf.cond(
                    tf.less(tf.random.uniform(()), 0.5)
                    and tf.less(scale[0], 1.0),
                    lambda: tf.random.uniform((), scale[0], 1.0),
                    lambda: tf.random.uniform(
                        (), tf.maximum(scale[0], 1.0), scale[1]
                    ),
                )
            coords = scale_coords(coords, sc)
            modified_coords = TFbT


if __name__ == "__main__":
    data = tf.ones([1, 1, 70, 83, 64])
    seg = tf.ones([1, 1, 70, 83, 64])
    patch_size = tf.cast([40, 56, 40], tf.int64)

    with tf.device("/CPU:0"):
        augment_spatial(data, seg, patch_size)
        # augment_spatial(data, seg, patch_size)
        # augment_spatial(data, seg, patch_size)
