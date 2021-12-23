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

# tf.debugging.set_log_device_placement(True)
from tfda.augmentations.utils import (
    create_zero_centered_coordinate_mesh,
    elastic_deform_coordinates,
    rotate_coords_2d,
    rotate_coords_3d,
    scale_coords
)
from tfda.base import TFT
from tfda.data_processing_utils import (
    center_crop_fn,
    interpolate_img,
    random_crop_fn,
    update_tf_channel
)
from tfda.utils import (
    TFbF,
    TFbT,
    TFf0,
    TFf1,
    TFi1,
    pi,
    to_tf_bool,
    to_tf_float
)

import tensorflow as tf

# Types
from typing import Optional


def augment_spatial_helper(sample_id: TFT, patch_size: TFT) -> TFT:
    pass


# @tf.function(experimental_follow_type_hints=True)
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

    # for sample_id in tf.range(data.shape[0]):
    def augment_per_sample(
        sample_id, patch_size, data, seg, data_result, seg_result
    ):
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

        # add from here
        if modified_coords:

            d = tf.constant(0)
            loop_cond = lambda d, coords: tf.less(d, dim)

            def body_fn(d, coords):
                # random crop always false
                # if random_crop:
                #     ctr = tf.random.uniform(
                #         [],
                #         patch_center_dist_from_border[d],
                #         tf.cast(tf.shape(data)[d + 2], dtype=tf.float32)
                #         - patch_center_dist_from_border[d],
                #     )
                # else:
                #     ctr = (
                #         tf.cast(tf.shape(data)[d + 2], dtype=tf.float32) / 2.0
                #         - 0.5
                #     )
                ctr = (
                    tf.cast(tf.shape(data)[d + 2], dtype=tf.float32) / 2.0
                    - 0.5
                )
                coords_d = coords[d] + ctr
                coords = update_tf_channel(coords, d, coords_d)
                d = d + 1
                coords.set_shape([3, None, None, None])
                return d, coords

            _, coords = tf.while_loop(
                loop_cond,
                body_fn,
                [d, coords],
                shape_invariants=[tf.TensorShape(None), coords.get_shape()],
            )
            data_sample = tf.zeros(tf.shape(data_result)[1:])
            channel_id = tf.constant(0)
            cond_to_loop_data = lambda channel_id, data_sample: tf.less(
                channel_id, tf.shape(data)[1]
            )

            def body_fn_data(channel_id, data_sample):
                data_channel = interpolate_img(
                    data[sample_id, channel_id],
                    coords,
                    order_data,
                    border_mode_data,
                    border_cval_data,
                )
                data_sample = update_tf_channel(
                    data_sample, channel_id, data_channel
                )
                channel_id = channel_id + 1
                return channel_id, data_sample

            _, data_sample = tf.while_loop(
                cond_to_loop_data, body_fn_data, [channel_id, data_sample]
            )
            data_result = update_tf_channel(
                data_result, sample_id, data_sample
            )
            if seg is not None:
                seg_sample = tf.zeros(tf.shape(seg_result)[1:])
                channel_id = tf.constant(0)
                cond_to_loop_seg = lambda channel_id, seg_sample: tf.less(
                    channel_id, tf.shape(seg)[1]
                )

                def body_fn_seg(channel_id, seg_sample):
                    seg_channel = interpolate_img(
                        seg[sample_id, channel_id],
                        coords,
                        order_seg,
                        border_mode_seg,
                        border_cval_seg,
                        is_seg=True,
                    )
                    seg_sample = update_tf_channel(
                        seg_sample, channel_id, seg_channel
                    )
                    channel_id = channel_id + 1
                    return channel_id, seg_sample

                _, seg_sample = tf.while_loop(
                    cond_to_loop_seg, body_fn_seg, [channel_id, seg_sample]
                )
                seg_result = update_tf_channel(
                    seg_result, sample_id, seg_sample
                )
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id : sample_id + 1]
            # if random_crop:
            #     # margin = [patch_center_dist_from_border[d] - tf.cast(patch_size[d], dtype=tf.float32) // 2 for d in range(dim)]
            #     margin = tf.map_fn(
            #         lambda d: tf.cast(
            #             patch_center_dist_from_border[d], dtype=tf.int64
            #         )
            #         - patch_size[d] // 2,
            #         elems=tf.range(dim),
            #     )
            #     d, s = random_crop_fn(
            #         data[sample_id : sample_id + 1], s, patch_size, margin
            #     )
            # else:
            #     d, s = center_crop_fn(
            #         data[sample_id : sample_id + 1], patch_size, s
            #     )
            d, s = center_crop_fn(
                data[sample_id : sample_id + 1], patch_size, s
            )
            data_result = update_tf_channel(data_result, sample_id, d[0])
            if seg is not None:
                seg_result = update_tf_channel(seg_result, sample_id, s[0])
        sample_id = sample_id + 1
        return sample_id, patch_size, data, seg, data_result, seg_result

    # spatial augment main body
    if not isinstance(patch_size, tf.Tensor):
        patch_size = tf.convert_to_tensor(patch_size)
    if not isinstance(patch_center_dist_from_border, tf.Tensor):
        patch_center_dist_from_border = tf.convert_to_tensor(
            patch_center_dist_from_border, dtype=tf.float32
        )
    cond_to_loop = lambda sample_id, patch_size, data, seg, data_result, seg_result: tf.less(
        sample_id, sample_num
    )
    dim = tf.shape(patch_size)[0]
    seg_result = None
    if seg is not None:
        seg_result = tf.cond(
            tf.equal(dim, tf.constant(2)),
            lambda: tf.zeros(
                tf.concat(
                    [tf.shape(seg, out_type=tf.int64)[:2], patch_size[:2]],
                    axis=0,
                )
            ),
            lambda: tf.zeros(
                tf.concat(
                    [tf.shape(seg, out_type=tf.int64)[:2], patch_size[:3]],
                    axis=0,
                )
            ),
        )

    data_result = tf.cond(
        tf.equal(dim, tf.constant(2)),
        lambda: tf.zeros(
            tf.concat(
                [tf.shape(data, out_type=tf.int64)[:2], patch_size[:2]], axis=0
            )
        ),
        lambda: tf.zeros(
            tf.concat(
                [tf.shape(data, out_type=tf.int64)[:2], patch_size[:3]], axis=0
            )
        ),
    )
    sample_num = tf.shape(data)[0]
    sample_id = tf.constant(0)
    _, _, _, _, data_result, seg_result = tf.while_loop(
        cond_to_loop,
        augment_per_sample,
        [sample_id, patch_size, data, seg, data_result, seg_result],
    )
    return data_result, seg_result


if __name__ == "__main__":
    data = tf.ones([1, 1, 70, 83, 64])
    seg = tf.ones([1, 1, 70, 83, 64])
    patch_size = tf.cast([40, 56, 40], tf.int64)
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # with tf.device("/CPU:0"):
    with mirrored_strategy.scope():
        augment_spatial(data, seg, patch_size, random_crop=TFbF)
        # augment_spatial(data, seg, patch_size)
        # augment_spatial(data, seg, patch_size)
