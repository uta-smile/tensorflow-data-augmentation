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
# Tensorflow
import tensorflow as tf

# tf.config.run_functions_eagerly(True)


@tf.function(experimental_follow_type_hints=True)
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


@tf.function(experimental_follow_type_hints=True)
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


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def augment_contrast(
    data_sample: tf.Tensor,
    contrast_range: tf.Tensor = (0.75, 1.25),
    preserve_range: tf.Tensor = True,
    per_channel: tf.Tensor = True,
    p_per_channel: tf.Tensor = 1.0,
) -> tf.Tensor:
    """Augment contrast."""

    return tf.cond(
        tf.logical_not(per_channel),
        lambda: tf.map_fn(
            lambda x: tf.cond(
                tf.less(tf.random.uniform(()), p_per_channel),
                lambda: augment_contrast_help(
                    x,
                    preserve_range,
                    augment_factor_help(
                        tf.less(tf.random.uniform(()), p_per_channel),
                        contrast_range,
                    ),
                ),
                lambda: x,
            ),
            data_sample,
        ),
        lambda: tf.map_fn(
            lambda x: tf.cond(
                tf.less(tf.random.uniform(()), p_per_channel),
                lambda: augment_contrast_help(
                    x,
                    preserve_range,
                    augment_factor_help(
                        tf.logical_and(
                            tf.less(tf.random.uniform(()), 0.5),
                            tf.less(contrast_range[0], 1),
                        ),
                        contrast_range,
                    ),
                ),
                lambda: x,
            ),
            data_sample,
        ),
    )

    # if tf.logical_not(per_channel):
    #     factor = augment_factor_help(
    #         tf.less(tf.random.uniform(()), p_per_channel), contrast_range
    #     )
    #     data_sample = tf.map_fn(
    #         lambda x: tf.cond(
    #             tf.less(tf.random.uniform(()), p_per_channel),
    #             lambda: augment_contrast_help(x, preserve_range, factor),
    #             lambda: x,
    #         ),
    #         data_sample,
    #     )
    # else:
    #     factor = augment_factor_help(
    #         _and(tf.less(tf.random.uniform(()), 0.5), tf.less(contrast_range[0], 1)),
    #         contrast_range,
    #     )
    #     data_sample = tf.map_fn(
    #         lambda x: tf.cond(
    #             tf.less(tf.random.uniform(()), p_per_channel),
    #             lambda: augment_contrast_help(x, preserve_range, factor),
    #             lambda: x,
    #         ),
    #         data_sample,
    #     )

    # return data_sample


@tf.function(experimental_follow_type_hints=True)
def augment_brightness_additive(
    data_sample: tf.Tensor,
    mu: tf.Tensor,
    sigma: tf.Tensor,
    per_channel: tf.Tensor = True,
    p_per_channel: tf.Tensor = 1.0,
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


@tf.function(experimental_follow_type_hints=True)
def augment_brightness_multiplicative(
    data_sample: tf.Tensor,
    multiplier_range: tf.Tensor = (0.5, 2),
    per_channel: tf.Tensor = True,
) -> tf.Tensor:
    """Augment brightness multiplicative."""
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


@tf.function(experimental_follow_type_hints=True)
def augment_gamma_help_h(data_sample: tf.Tensor, new_data_sample) -> tf.Tensor:
    mn = tf.math.reduce_mean(data_sample)
    sd = tf.math.reduce_std(data_sample)

    data_sample = new_data_sample - tf.math.reduce_mean(new_data_sample)
    data_sample = data_sample / (tf.math.reduce_std(data_sample) + 1e-8) * sd
    return data_sample + mn


@tf.function(experimental_follow_type_hints=True)
def augment_gamma_help(
    data_sample: tf.Tensor,
    gamma_range: tf.Tensor,
    epsilon: tf.Tensor,
    retain_stats: tf.Tensor,
) -> tf.Tensor:
    gamma = tf.cond(
        tf.logical_and(
            tf.less(tf.random.uniform(()), 0.5), tf.less(gamma_range[0], 1)
        ),
        lambda: tf.random.uniform((), minval=gamma_range[0], maxval=1),
        lambda: tf.random.uniform(
            (), minval=tf.maximum(gamma_range[0], 1), maxval=gamma_range[1]
        ),
    )
    minm = tf.math.reduce_min(data_sample)
    rnge = tf.math.reduce_max(data_sample) - minm

    new_data_sample = (
        tf.math.pow(
            ((data_sample - minm) / tf.cast(rnge + epsilon, dtype=tf.float32)),
            gamma,
        )
        * rnge
        + minm
    )

    return tf.cond(
        retain_stats,
        lambda: augment_gamma_help_h(data_sample, new_data_sample),
        lambda: new_data_sample,
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    ]
)
def augment_gamma(
    data_sample: tf.Tensor,
    gamma_range: tf.Tensor = (0.5, 2),
    invert_image: tf.Tensor = False,
    epsilon: tf.Tensor = 1e-7,
    per_channel: tf.Tensor = True,
    retain_stats: tf.Tensor = True,
):
    data_sample = tf.cond(
        invert_image,
        lambda: -data_sample,
        lambda: data_sample,
    )
    data_sample = tf.cond(
        tf.logical_not(per_channel),
        lambda: augment_gamma_help(
            data_sample, gamma_range, epsilon, retain_stats
        ),
        lambda: tf.map_fn(
            lambda ds: augment_gamma_help(
                ds, gamma_range, epsilon, retain_stats
            ),
            data_sample,
        ),
    )
    data_sample = tf.cond(
        invert_image,
        lambda: -data_sample,
        lambda: data_sample,
    )
    return data_sample


def test():
    """Test."""
    # dataset = (
    #     tf.data.Dataset.range(10, output_type=tf.float32).batch(5).batch(2)
    # )
    # data_sample = next(iter(dataset))[0]
    data = tf.random.uniform((1, 320, 320))
    mu = tf.cast(0, tf.float32)
    sigma = tf.cast(0.1, tf.float32)

    with tf.device("/CPU:0"):
        # https://github.com/tensorflow/tensorflow/issues/49202
        tf.print(augment_gamma(data).shape)
        # tf.print(augment_contrast(data_sample).shape)
        # tf.print(augment_brightness_additive(data_sample, mu, sigma).shape)
        # tf.print(augment_brightness_multiplicative(data_sample).shape)


if __name__ == "__main__":
    test()
