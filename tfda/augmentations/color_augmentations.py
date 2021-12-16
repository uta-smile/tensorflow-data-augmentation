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
filename : color_augmentations.py
project  : augmentations
license  : GPL-3.0+

Color Augmentation
"""
import tensorflow as tf

# Others
from tfda.utils import TFbT, TFf1


@tf.function
def augment_contrast_help(
    x: tf.Tensor, preserve_range: tf.Tensor, factor: tf.Tensor
) -> tf.Tensor:
    """Augment contrast help function."""
    mn = tf.math.reduce_mean(x)
    nx = (x - mn) * factor + mn
    if preserve_range:
        minm = tf.math.reduce_min(x)
        maxm = tf.math.reduce_max(x)

        nx = tf.where(nx < minm, minm, nx)
        nx = tf.where(nx > maxm, maxm, nx)
    return nx


@tf.function
def augment_factor_help(
    pred: tf.Tensor, contrast_range: tf.Tensor
) -> tf.Tensor:
    """Augment factor help."""
    mi, ma = contrast_range[0], contrast_range[1]
    return tf.cond(
        pred,
        lambda: tf.random.uniform((), mi, 1),
        lambda: tf.random.uniform((), tf.math.maximum(mi, 1), ma),
    )


@tf.function
def augment_contrast(
    data_sample: tf.Tensor,
    contrast_range: tf.Tensor = None,
    preserve_range: tf.Tensor = TFbT,
    per_channel: tf.Tensor = TFbT,
    p_per_channel: tf.Tensor = TFf1,
) -> tf.Tensor:
    """Augment contrast."""
    if contrast_range is None:
        contrast_range = tf.cast([0.75, 1.25], tf.float32)
    # TODO: callable contrast_range
    if not per_channel:
        factor = augment_factor_help(
            tf.random.uniform(()) < p_per_channel, contrast_range
        )
        data_sample = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < p_per_channel,
                lambda: augment_contrast_help(x, preserve_range, factor),
                lambda: x,
            ),
            data_sample,
        )

    else:
        factor = augment_factor_help(
            tf.random.uniform(()) < 0.5 and contrast_range[0] < 1,
            contrast_range,
        )
        data_sample = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < p_per_channel,
                lambda: augment_contrast_help(x, preserve_range, factor),
                lambda: x,
            ),
            data_sample,
        )

    return data_sample


@tf.function
def augment_brightness_additive(
    data_sample: tf.Tensor,
    mu: tf.Tensor,
    sigma: tf.Tensor,
    per_channel: tf.Tensor = TFbT,
    p_per_channel: tf.Tensor = TFf1,
) -> tf.Tensor:
    """Augment brightness additive."""
    if not per_channel:
        rnd_nb = tf.random.uniform((), mu, sigma)
        data_sample = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < p_per_channel,
                lambda: x + rnd_nb,
                lambda: x,
            ),
            data_sample,
        )
    else:
        data_sample = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < p_per_channel,
                lambda: x + tf.random.uniform((), mu, sigma),
                lambda: x,
            ),
            data_sample,
        )
    return data_sample


@tf.function
def augment_brightness_multiplicative(
    data_sample: tf.Tensor,
    multiplier_range: tf.Tensor = None,
    per_channel: tf.Tensor = TFbT,
) -> tf.Tensor:
    """Augment brightness multiplicative."""
    if multiplier_range is None:
        multiplier_range = tf.cast([0.5, 2], tf.float32)
    multiplier = tf.random.uniform(
        (), multiplier_range[0], multiplier_range[1]
    )

    if not per_channel:
        data_sample *= multiplier
    else:
        data_sample = tf.map_fn(
            lambda x: x
            * tf.random.uniform((), multiplier_range[0], multiplier_range[1]),
            data_sample,
        )
    return data_sample


if __name__ == "__main__":
    dataset = (
        tf.data.Dataset.range(10, output_type=tf.float32).batch(5).batch(2)
    )
    data_sample = next(iter(dataset))[0]
    mu = tf.cast(0, tf.float32)
    sigma = tf.cast(0.1, tf.float32)

    with tf.device("/CPU:0"):
        # https://github.com/tensorflow/tensorflow/issues/49202
        tf.print(augment_contrast(data_sample).shape)
        tf.print(augment_brightness_additive(data_sample, mu, sigma).shape)
        tf.print(augment_brightness_multiplicative(data_sample).shape)
