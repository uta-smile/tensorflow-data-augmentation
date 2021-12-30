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

from typing import Tuple
import tensorflow as tf

# Others
# tf.debugging.set_log_device_placement(True)
from tfda.augmentations.utils import gaussian_filter, get_range_val
from tfda.base import TFT
from tfda.utils import TFbF, TFbT, TFf1, nan, to_tf_float


@tf.function(experimental_follow_type_hints=True)
def gn_var_fn(noise_variance: Tuple[float, float]) -> tf.Tensor:
    """Gaussian noise variance fn."""
    # TODO: solve Skipping loop optimization issue
    # mean, std = noise_variance
    # mean, std = tf.constant(mean, tf.float32), tf.constant(std, tf.float32)
    # return tf.cond(
    #     tf.equal(mean, std),
    #     lambda: mean,
    #     lambda: tf.random.uniform((), mean, std),
    # )
    return tf.random.uniform((), noise_variance[0], noise_variance[1])


@tf.function(experimental_follow_type_hints=True)
def augment_gaussian_noise(
    data_sample: tf.Tensor,
    noise_variance: Tuple[float, float] = (0, 0.1),
    p_per_channel: tf.Tensor = TFf1,
    per_channel: tf.Tensor = TFbF,
) -> tf.Tensor:
    """Apply gaussian noise on tf Tensor."""
    # TODO: solve Skipping loop optimization issue
    # variance = tf.cond(
    #     not per_channel,
    #     lambda: gn_var_fn(noise_variance),
    #     lambda: nan,
    # )
    variance = gn_var_fn(noise_variance)

    return tf.map_fn(
        lambda x: tf.cond(
            tf.random.uniform(()) < p_per_channel,
            lambda: x
            + tf.random.normal(
                tf.shape(x),
                0,
                tf.cond(
                    not tf.reduce_any(tf.math.is_nan(variance)),
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
    data_sample: TFT,
    sigma_range: TFT,
    per_channel: TFT = TFbT,
    p_per_channel: TFT = 1.0,
    # TODO: not used
    different_sigma_per_axis: TFT = TFbF,
    p_isotropic: TFT = 0.0,
):
    sigma = get_range_val(sigma_range)

    return tf.map_fn(
        lambda x: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: gaussian_filter(
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
        tf.print(augment_gaussian_noise(datasample))
        # tf.print(gn_var_fn((0, 0.1)))
