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

# Tensorflow
import tensorflow as tf

# Local
# tf.debugging.set_log_device_placement(True)
from tfda.augmentations.utils import gaussian_filter, gaussian_filter_2D, get_range_val
from tfda.defs import nan
from tfda.utils import isnotnan


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.float32)])
def gn_var_fn(noise_variance: tf.Tensor) -> tf.Tensor:
    """Gaussian noise variance fn."""
    mean, std = noise_variance[0], noise_variance[1]
    return tf.cond(
        tf.equal(mean, std),
        lambda: mean,
        lambda: tf.random.uniform((), mean, std),
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    ]
)
def augment_gaussian_noise(
    data_sample: tf.Tensor,
    noise_variance: tf.Tensor = (0, 0.1),
    p_per_channel: tf.Tensor = 1.0,
    per_channel: tf.Tensor = False,
) -> tf.Tensor:
    """Apply gaussian noise on tf Tensor."""
    variance = tf.cond(
        tf.logical_not(per_channel),
        lambda: gn_var_fn(noise_variance),
        lambda: nan,
    )

    return tf.map_fn(
        lambda x: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: x
            + tf.random.normal(
                tf.shape(x),
                0,
                tf.cond(
                    isnotnan(variance),
                    lambda: variance,
                    lambda: gn_var_fn(noise_variance),
                ),
            ),
            lambda: x,
        ),
        data_sample,
    )


@tf.function(experimental_follow_type_hints=True)
def augment_gaussian_blur(
    data_sample: tf.Tensor,
    sigma_range: tf.Tensor,
    per_channel: tf.Tensor = True,
    p_per_channel: tf.Tensor = 1.0,
    # TODO: not used
    different_sigma_per_axis: tf.Tensor = False,
    p_isotropic: tf.Tensor = 0.0,
):
    sigma = get_range_val(sigma_range)

    return tf.map_fn(
        lambda x: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: gaussian_filter_2D(
                x,
                sigma=tf.cond(
                    per_channel,
                    lambda: get_range_val(sigma_range),
                    lambda: sigma,
                ),
            ),
            lambda: x,
        ),
        data_sample,
    )

@tf.function(experimental_follow_type_hints=True)
def augment_gaussian_blur_2D(
    data_sample: tf.Tensor,
    sigma_range: tf.Tensor,
    per_channel: tf.Tensor = True,
    p_per_channel: tf.Tensor = 1.0,
    # TODO: not used
    different_sigma_per_axis: tf.Tensor = False,
    p_isotropic: tf.Tensor = 0.0,
):
    sigma = get_range_val(sigma_range)

    return tf.map_fn(
        lambda x: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: gaussian_filter_2D(
                x,
                sigma=tf.cond(
                    per_channel,
                    lambda: get_range_val(sigma_range),
                    lambda: sigma,
                ),
            ),
            lambda: x,
        ),
        data_sample,
    )


if __name__ == "__main__":
    dataset = (
        tf.data.Dataset.range(10, output_type=tf.float32).batch(5).batch(2)
    )
    xs = next(iter(dataset))
    datasample = tf.cast(xs[0], tf.float32)

    with tf.device("/CPU:0"):
        # https://github.com/tensorflow/tensorflow/issues/49202
        tf.print(gn_var_fn([0, 0.1]))
        tf.print(augment_gaussian_noise(datasample))
