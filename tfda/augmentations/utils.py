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

# tf.debugging.set_log_device_placement(True)
from tfda.base import TFT
from tfda.utils import to_tf_float, to_tf_int, TFbF, TFbT, TFf0


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
    coords = tf.cast(
        tf.meshgrid(tmp[0], tmp[1], tmp[2], indexing="ij"), dtype=tf.float32
    )

    coords = tf.map_fn(
        lambda i: coords[to_tf_int(i)]
        - ((to_tf_float(shape) - 1) / 2)[to_tf_int(i)],
        tf.range(coords.shape[0], dtype=tf.float32),
    )
    return coords


# Gaussian filter related


@tf.function
def gaussian_kernel1d(sigma: TFT, radius: TFT) -> TFT:
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    phi = tf.exp(-0.5 / (sigma * sigma) * x ** 2)
    return phi / tf.reduce_sum(phi)


@tf.function
def gaussian_filter1d(input, sigma):
    lw = tf.cast(sigma * sigma + 0.5, tf.int64)
    weights = gaussian_kernel1d(sigma, lw)[::-1]

    input = tf.reshape(input, (1, -1, 1))
    kernel = tf.reshape(weights, (-1, 1, 1))

    return tf.squeeze(tf.nn.conv1d(input, kernel, stride=1, padding="SAME"))


@tf.function
def gaussian_filter(
    input: TFT, sigma: TFT, mode: str = "reflect", cavl: TFT = TFf0
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
                            lambda x: gaussian_filter1d(x, sigma), xs
                        ),
                        tf.transpose(gfa, perm),
                    ),
                    input.shape,
                ),
                trans,
                input,
            ),
            (input.shape[0], input.shape[2], input.shape[1]),
        ),
        (1, 2, 0),
    )


if __name__ == "__main__":
    import scipy.ndimage.filters as sf

    patch_size = tf.constant([40, 56, 40])

    with tf.device("/CPU:0"):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        xs = tf.random.uniform(coords.shape[1:], 0, 1)
        x = gaussian_filter(xs, 5)
        x_ = sf.gaussian_filter(xs, 5, mode="constant")
        tf.print(x[0][0], "\n", x.shape, x[0].shape)
        tf.print("----\n", x_[0][0], "\n", x_.shape, x_[0].shape)
