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
filename : utils.py
project  : augmentations
license  : GPL-3.0+

Augmentation Utils
"""

import tensorflow as tf

# Local
from tfda.defs import TFbF, TFbT, nan
from tfda.utils import isnan, to_tf_bool, to_tf_float, to_tf_int

# tf.debugging.set_log_device_placement(True)


@tf.function(experimental_follow_type_hints=True)
def to_one_hot(seg: tf.Tensor, all_seg_labels: tf.Tensor = nan):
    if isnan(all_seg_labels):
        all_seg_labels, _ = tf.unique(tf.reshape(seg, (-1,)))

    nseg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for s in tf.range(tf.shape(seg)[0]):
        result = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in tf.range(tf.size(all_seg_labels)):
            result = result.write(
                i, tf.where(seg[s] == all_seg_labels[i], 1.0, 0.0)
            )
        nseg = nseg.write(s, result.stack())
    return nseg.stack()


@tf.function(experimental_follow_type_hints=True)
def get_range_val(value: tf.Tensor, rnd_type: tf.Tensor = "uniform"):
    # TODO: different rank values

    # return tf.case(
    #     [
    #         (tf.equal(tf.rank(value), 0), lambda: value),
    #         (tf.equal(tf.shape(value)[0], 1), lambda: value[0]),
    #         (tf.equal(value[0], value[1]), lambda: value[0]),
    #         (
    #             tf.equal(tf.shape(value)[0], 2) and tf.equal(rnd_type, "uniform"),
    #             lambda: tf.random.uniform(
    #                 (), minval=value[0], maxval=value[1], dtype=tf.float32
    #             ),
    #         ),
    #         (
    #             tf.equal(rnd_type, "normal"),
    #             lambda: tf.random.normal(
    #                 (), mean=value[0], stddev=value[1], dtype=tf.float32
    #             ),
    #         ),
    #     ],
    #     (lambda: value)
    # )
    return tf.case(
        [
            (
                tf.equal(tf.shape(value)[0], 2)
                and tf.equal(rnd_type, "uniform"),
                lambda: tf.random.uniform(
                    (), minval=value[0], maxval=value[1], dtype=tf.float32
                ),
            ),
            (
                tf.equal(rnd_type, "normal"),
                lambda: tf.random.normal(
                    (), mean=value[0], stddev=value[1], dtype=tf.float32
                ),
            ),
        ]
    )


@tf.function(experimental_follow_type_hints=True)
def create_zero_centered_coordinate_mesh(shape: tf.Tensor) -> tf.Tensor:
    tmp = tf.map_fn(
        lambda x: tf.range(x, dtype=tf.float32),
        shape,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=[None], dtype=tf.float32
        ),
    )

    # TODO: change hardcode to others
    # How to use *tmp in tensorflow graph?
    # tf.meshgrid(*tmp, indexing="ij")
    coords = tf.cast(
        tf.meshgrid(tmp[0], tmp[1], tmp[2], indexing="ij"), dtype=tf.float32
    )

    return tf.map_fn(
        lambda i: coords[to_tf_int(i)]
        - ((to_tf_float(shape) - 1) / 2)[to_tf_int(i)],
        tf.range(tf.shape(coords)[0], dtype=tf.float32),
    )


@tf.function(experimental_follow_type_hints=True)
def elastic_deform_coordinates(
    coordinates: tf.Tensor, alpha: tf.Tensor, sigma: tf.Tensor
):
    coordinates = tf.cast(coordinates, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    return (
        tf.map_fn(
            lambda _: gaussian_filter(
                (tf.random.uniform(tf.shape(coordinates)[1:]) * 2 - 1),
                sigma,
                mode="constant",
            )
            * alpha,
            tf.range(tf.shape(coordinates)[0], dtype=tf.float32),
        )
        + coordinates
    )


# rotation relate
@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_x_3d(
    angle: tf.Tensor, matrix: tf.Tensor
) -> tf.Tensor:
    rotation_x = tf.cast(
        [
            [1, 0, 0],
            [0, tf.cos(angle), -tf.sin(angle)],
            [0, tf.sin(angle), tf.cos(angle)],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_y_3d(
    angle: tf.Tensor, matrix: tf.Tensor
) -> tf.Tensor:
    rotation_x = tf.cast(
        [
            [tf.cos(angle), 0, tf.sin(angle)],
            [0, 1, 0],
            [-tf.sin(angle), 0, tf.cos(angle)],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_z_3d(
    angle: tf.Tensor, matrix: tf.Tensor
) -> tf.Tensor:
    rotation_x = tf.cast(
        [
            [tf.cos(angle), -tf.sin(angle), 0],
            [tf.sin(angle), tf.cos(angle), 0],
            [0, 0, 1],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_2d(
    angle: tf.Tensor, matrix: tf.Tensor = None
) -> tf.Tensor:
    rotation = tf.cast(
        [[tf.cos(angle), -tf.sin(angle)], [tf.sin(angle), tf.cos(angle)]],
        tf.float32,
    )

    if matrix is None:
        return rotation
    return matrix @ rotation
    # return rotation


@tf.function(experimental_follow_type_hints=True)
def rotate_coords_3d(
    coords: tf.Tensor,
    angle_x: tf.Tensor,
    angle_y: tf.Tensor,
    angle_z: tf.Tensor,
) -> tf.Tensor:
    rot_matrix = tf.eye(tf.shape(coords)[0])

    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)

    return tf.reshape(
        tf.transpose(
            tf.transpose(tf.reshape(coords, (tf.shape(coords)[0], -1)))
            @ rot_matrix
        ),
        tf.shape(coords),
    )


@tf.function(experimental_follow_type_hints=True)
def rotate_coords_2d(coords: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    rot_matrix = create_matrix_rotation_2d(angle)
    print(tf.shape(coords))
    return tf.reshape(
        tf.transpose(
            tf.transpose(tf.reshape(coords, (tf.shape(coords)[0], -1)))
            @ rot_matrix
        ),
        tf.shape(coords),
    )


@tf.function(experimental_follow_type_hints=True)
def scale_coords(coords: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    return coords * tf.reshape(scale, (-1, 1))


# Gaussian filter related
@tf.function(experimental_follow_type_hints=True)
def gaussian_kernel1d(sigma: tf.Tensor, radius: tf.Tensor) -> tf.Tensor:
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    phi = tf.exp(-0.5 / (sigma * sigma) * x ** 2)
    return phi / tf.reduce_sum(phi)


@tf.function(experimental_follow_type_hints=True)
def gf_pad(
    x: tf.Tensor,
    mode: tf.Tensor,
    cval: tf.Tensor,
    ws: tf.Tensor,
    ins: tf.Tensor,
):
    """Pad.

    for the VALID, width = (pin_w - k_w + 1) / stride(1) = in_w
    padding size = pin_w - in_w = k_w - 1
    Left geq Right
    """
    pv = tf.cond(
        tf.equal(mode, "reflect"),
        lambda: x,
        lambda: tf.zeros(ins) + cval,
    )
    lp = tf.concat([pv, pv[::-1]], axis=0)
    rp = tf.concat([pv[::-1], pv], axis=0)

    ll = (ws - 1 + 1) // 2
    rl = (ws - 1) - ll

    lpv = tf.tile(lp, [ll // 2 // ins + 1])[:ll][::-1]
    lrv = tf.tile(rp, [rl // 2 // ins + 1])[:rl]
    return tf.concat([lpv, x, lrv], axis=0)


@tf.function(experimental_follow_type_hints=True)
def gaussian_filter1d(
    xs: tf.Tensor,
    sigma: tf.Tensor,
    mode: tf.Tensor,
    cval: tf.Tensor = 0.0,
):
    sigma = tf.cast(sigma, tf.float32)
    lw = tf.cast(sigma * sigma + 0.5, tf.int64)
    weights = gaussian_kernel1d(sigma, lw)[::-1]

    ws = tf.size(weights)
    ins = tf.shape(xs)[1]

    # padding
    xs = tf.map_fn(
        lambda x: gf_pad(
            x,
            mode,
            cval,
            ws,
            ins,
        ),
        xs,
    )

    xs = tf.reshape(xs, (-1, tf.shape(xs)[1], 1))
    kernel = tf.reshape(weights, (-1, 1, 1))

    return tf.squeeze(tf.nn.conv1d(xs, kernel, stride=1, padding="VALID"))


@tf.function(experimental_follow_type_hints=True)
def gaussian_filter(
    xxs: tf.Tensor,
    sigma: tf.Tensor,
    mode: tf.Tensor = "reflect",
    cval: tf.Tensor = 0.0,
) -> tf.Tensor:
    """Gaussian filter trans from scipy gaussian filter.

    NOTE: only for 3 dim Tensor.
    """

    # NOTE: useless in tf
    # orders = tf.zeros(input.ndim)
    # sigmas = tf.repeat(sigma, input.ndim)
    # modes = tf.repeat(tf.cast(mode, tf.string), input.ndim)
    # output = tf.zeros(input.shape, dtype=tf.float32)
    # axes = tf.range(input.shape[0])

    # trans = tf.cast([[0, 1, 2], [2, 1, 0], [0, 2, 1]], tf.int64)

    perms = tf.constant([[2, 1, 0], [2, 0, 1], [1, 0, 2]])
    rperms = tf.constant([[2, 1, 0], [1, 2, 0], [1, 0, 2]])

    return tf.foldl(
        lambda gfa, idx: tf.reshape(
            tf.transpose(
                tf.map_fn(
                    lambda xs: gaussian_filter1d(xs, sigma, mode, cval),
                    tf.transpose(gfa, perms[idx]),
                ),
                rperms[idx],
            ),
            tf.shape(xxs),
        ),
        tf.range(3),
        xxs,
    )


if __name__ == "__main__":
    # Others
    import scipy.ndimage.filters as sf

    patch_size = tf.constant([20, 376, 376])

    with tf.device("/CPU:0"):
        tf.print(get_range_val([0, 1.0]))
        coords = create_zero_centered_coordinate_mesh(patch_size)

        tf.print(elastic_deform_coordinates(coords, 50.0, 12.0).shape)
        # assert elastic_deform_coordinates(coords, 50., 12.).shape == [
        #     3,
        #     40,
        #     56,
        #     40,
        # ]

        xs = tf.random.uniform(patch_size, 0, 1)
        s = xs.shape
        tf.print(xs.shape)
        x = gaussian_filter(xs, 5, "reflect")
        tf.print("\n\n", x[0][0], "\n", x.shape, x[0].shape)
        x_ = sf.gaussian_filter(xs, 5, mode="reflect")
        tf.print("----\n", x_[0][0], "\n", x_.shape, x_[0].shape)

        tf.print("----------")
        tf.print(rotate_coords_3d(coords, 1.0, 1.0, 1.0).shape)

        tf.print(to_one_hot(tf.zeros([9, 40, 56, 40]), [1.0, 2, 3]).shape)
