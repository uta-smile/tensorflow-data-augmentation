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
date     : Dec 10, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : noise_augmentations.py
project  : augmentations
license  : GPL-3.0+

Noise Augmentations
"""

# Standard Library
from tfda.base import TFT


import tensorflow as tf

# Others
from tfda.utils import TFbF, TFf1, to_tf_bool


@tf.function
def gn_var_fn(noise_variance: TFT) -> tf.Tensor:
    """Gaussian noise variance fn."""
    return tf.cond(
        to_tf_bool(noise_variance[0] == noise_variance[1]),
        lambda: tf.cast(noise_variance[0], dtype=tf.float32),
        lambda: tf.random.uniform((), noise_variance[0], noise_variance[1]),
    )


@tf.function
def augment_gaussian_noise(
    data_sample: tf.Tensor,
    noise_variance: tf.Tensor = (0, 0.1),
    p_per_channel: tf.Tensor = TFf1,
    per_channel: tf.bool = TFbF,
) -> tf.Tensor:
    """Apply gaussian noise on tf Tensor."""
    variance = tf.cond(
        to_tf_bool(not per_channel),
        lambda: gn_var_fn(noise_variance),
        lambda: tf.cast([], tf.float32),
    )

    return tf.map_fn(
        lambda x: tf.cond(
            tf.random.uniform(()) < p_per_channel,
            lambda: x
            + tf.random.normal(
                x.shape,
                0,
                tf.cond(
                    to_tf_bool(variance is not None),
                    lambda: variance,
                    lambda: gn_var_fn(noise_variance),
                ),
            ),
            lambda: x,
        ),
        data_sample,
    )


if __name__ == "__main__":
    dataset = tf.data.Dataset.range(10, output_type=tf.float32).batch(5).batch(2)
    xs = next(iter(dataset))
    datasample = tf.cast(xs[0], tf.float32)

    with tf.device("/CPU:0"):
        # https://github.com/tensorflow/tensorflow/issues/49202
        print(augment_gaussian_noise(datasample))
