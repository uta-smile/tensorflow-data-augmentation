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

# Others
# tf.debugging.set_log_device_placement(True)
from tfda.base import TFT
from tfda.utils import TFbF, TFbT, TFf0, to_tf_bool, to_tf_float, to_tf_int


@tf.function
def get_range_val(value):
    if tf.equal(value[0], value[1]):
        n_val = value[0]
    else:
        n_val = tf.random.uniform((), minval=value[0], maxval=value[1], dtype=tf.float32)
    return n_val


@tf.function
def create_zero_centered_coordinate_mesh(shape: TFT) -> TFT:
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
        tf.range(coords.shape[0], dtype=tf.float32),
    )


@tf.function
def elastic_deform_coordinates(coordinates: TFT, alpha: TFT, sigma: TFT):
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
@tf.function
def create_matrix_rotation_x_3d(angle: TFT, matrix: TFT) -> TFT:
    rotation_x = tf.cast(
        [
            [1, 0, 0],
            [0, tf.cos(angle), -tf.sin(angle)],
            [0, tf.sin(angle), tf.cos(angle)],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function
def create_matrix_rotation_y_3d(angle: TFT, matrix: TFT) -> TFT:
    rotation_x = tf.cast(
        [
            [tf.cos(angle), 0, tf.sin(angle)],
            [0, 1, 0],
            [-tf.sin(angle), 0, tf.cos(angle)],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function
def create_matrix_rotation_z_3d(angle: TFT, matrix: TFT) -> TFT:
    rotation_x = tf.cast(
        [
            [tf.cos(angle), -tf.sin(angle), 0],
            [tf.sin(angle), tf.cos(angle), 0],
            [0, 0, 1],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function
def create_matrix_rotation_2d(angle: TFT, matrix: TFT = None) -> TFT:
    rotation = tf.cast(
        [[tf.cos(angle), -tf.sin(angle)], [tf.sin(angle), tf.cos(angle)]],
        tf.float32,
    )

    if matrix is None:
        return rotation
    return matrix @ rotation
    # return rotation


@tf.function
def rotate_coords_3d(
    coords: TFT, angle_x: TFT, angle_y: TFT, angle_z: TFT
) -> TFT:
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
def rotate_coords_2d(coords: TFT, angle: TFT) -> TFT:
    rot_matrix = create_matrix_rotation_2d(angle)
    print(tf.shape(coords))
    return tf.reshape(
        tf.transpose(
            tf.transpose(tf.reshape(coords, (tf.shape(coords)[0], -1)))
            @ rot_matrix
        ),
        tf.shape(coords),
    )


@tf.function
def scale_coords(coords: TFT, scale: TFT) -> TFT:
    return coords * tf.reshape(scale, (-1, 1))


# Gaussian filter related
@tf.function(experimental_follow_type_hints=True)
def gaussian_kernel1d(sigma: TFT, radius: TFT) -> TFT:
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    phi = tf.exp(-0.5 / (sigma * sigma) * x ** 2)
    return phi / tf.reduce_sum(phi)


@tf.function(experimental_follow_type_hints=True)
def gaussian_filter1d(input: TFT, sigma: TFT, mode: TFT, cval: TFT = TFf0):
    sigma = tf.cast(sigma, tf.float32)
    lw = tf.cast(sigma * sigma + 0.5, tf.int64)
    weights = gaussian_kernel1d(sigma, lw)[::-1]

    ws = tf.size(weights)
    ins = tf.size(input)

    # padding
    # for the VALID, width = (pin_w - k_w + 1) / stride(1) = in_w
    # padding size = pin_w - in_w = k_w - 1
    # Left geq Right
    pv = tf.cond(
        to_tf_bool(mode == "reflect"),
        lambda: input,
        lambda: tf.zeros(ins) + cval,
    )
    lp = tf.concat([pv, pv[::-1]], axis=0)
    rp = tf.concat([pv[::-1], pv], axis=0)

    ll = (ws - 1 + 1) // 2
    rl = (ws - 1) - ll

    lpv = tf.tile(lp, [ll // 2 // ins + 1])[:ll][::-1]
    lrv = tf.tile(rp, [rl // 2 // ins + 1])[:rl]
    input = tf.concat([lpv, input, lrv], axis=0)

    input = tf.reshape(input, (1, -1, 1))
    kernel = tf.reshape(weights, (-1, 1, 1))

    return tf.squeeze(tf.nn.conv1d(input, kernel, stride=1, padding="VALID"))


@tf.function(experimental_follow_type_hints=True)
def gaussian_filter(
    input: TFT, sigma: TFT, mode: str = "reflect", cval: TFT = TFf0
) -> TFT:
    """Gaussian filter trans from scipy gaussian filter."""

    # NOTE: useless in tf
    # orders = tf.zeros(input.ndim)
    # sigmas = tf.repeat(sigma, input.ndim)
    # modes = tf.repeat(tf.cast(mode, tf.string), input.ndim)
    # output = tf.zeros(input.shape, dtype=tf.float32)
    # axes = tf.range(input.shape[0])

    # TF graph failed
    # trans = tf.cast([[2, 1, 0], [2, 0, 1], [0, 1, 2]], tf.int64)
    # rtrans = tf.cast([[2, 1, 0], [1, 2, 0], [0, 1, 2]], tf.int64)
    # return tf.foldl(
    #     lambda gfa, i: tf.transpose(
    #         tf.map_fn(
    #             lambda xs: tf.map_fn(
    #                 lambda x: gaussian_filter1d(x, sigma), xs
    #             ),
    #             tf.transpose(gfa, trans[i]),
    #         ),
    #         rtrans[i],
    #     ),
    #     tf.range(3),
    #     input,
    # )

    trans = tf.cast([[0, 1, 2], [2, 1, 0], [0, 2, 1]], tf.int64)
    return tf.transpose(
        tf.reshape(
            tf.foldl(
                lambda gfa, perm: tf.reshape(
                    tf.map_fn(
                        lambda xs: tf.map_fn(
                            lambda x: gaussian_filter1d(
                                tf.reshape(x, (-1,)),
                                sigma,
                                tf.cast(mode, tf.string),
                                cval,
                            ),
                            xs,
                        ),
                        tf.transpose(gfa, perm),
                    ),
                    tf.shape(input),
                ),
                trans,
                input,
            ),
            (tf.shape(input)[0], tf.shape(input)[2], tf.shape(input)[1]),
        ),
        (1, 2, 0),
    )


if __name__ == "__main__":
    # Others
    import scipy.ndimage.filters as sf

    patch_size = tf.constant([40, 56, 40])

    with tf.device("/CPU:0"):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        tf.print(elastic_deform_coordinates(coords, 50, 12).shape)
        assert elastic_deform_coordinates(coords, 50, 12).shape == [
            3,
            40,
            56,
            40,
        ]

        xs = tf.random.uniform(patch_size, 0, 1)
        s = xs.shape
        tf.print(xs.shape)
        x = gaussian_filter(xs, 5, "reflect")
        x_ = sf.gaussian_filter(xs, 5, mode="reflect")
        tf.print("\n\n", x[0][0], "\n", x.shape, x[0].shape)
        tf.print("----\n", x_[0][0], "\n", x_.shape, x_[0].shape)

        tf.print("----------")
        tf.print(rotate_coords_3d(coords, 1.0, 1.0, 1.0).shape)
