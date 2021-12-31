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
date     : Dec  3, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : base.py
project  : tfda

Tensorflow data augmentation base
"""
# Standard Library

import tensorflow as tf

# Types
from typing import Any

# Local
# tf.debugging.set_log_device_placement(True)
from tfda.defs import TFDADefs, nan


class TFDABase(tf.keras.layers.Layer):
    """Tensorflow data augmentation base."""

    def __init__(
        self,
        data_key: str = "data",
        label_key: str = "seg",
        p_per_sample: tf.Tensor = 1.0,
        p_per_channel: tf.Tensor = 1.0,
        per_channel: tf.Tensor = False,
        contrast_range: tf.Tensor = (0.75, 1.25),
        multiplier_range: tf.Tensor = (0.5, 2),
        preserve_range: tf.Tensor = True,
        noise_variance: tf.Tensor = (0.0, 0.1),
        different_sigma_per_channel: tf.Tensor = True,
        gamma_range: tf.Tensor = (0.5, 2),
        invert_image: tf.Tensor = False,
        retain_stats: tf.Tensor = False,
        blur_sigma: tf.Tensor = (1.0, 5.0),
        zoom_range: tf.Tensor = (0.5, 1.0),
        order_downsample: tf.Tensor = 1,
        order_upsample: tf.Tensor = 0,
        ignore_axes: tf.Tensor = nan,
        **kws,
    ) -> None:
        super().__init__(**kws)
        self.defs = TFDADefs(
            p_per_sample=p_per_sample,
            p_per_channel=p_per_channel,
            per_channel=per_channel,
            contrast_range=contrast_range,
            multiplier_range=multiplier_range,
            preserve_range=preserve_range,
            noise_variance=noise_variance,
            different_sigma_per_channel=different_sigma_per_channel,
            gamma_range=gamma_range,
            invert_image=invert_image,
            retain_stats=retain_stats,
            blur_sigma=blur_sigma,
            zoom_range=zoom_range,
            order_upsample=order_upsample,
            order_downsample=order_downsample,
            ignore_axes=ignore_axes,
            # not tensor
            data_key=data_key,
            label_key=label_key,
        )


Compose = tf.keras.Sequential


if __name__ == "__main__":
    pass
