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
filename : noise_transforms.py
project  : transforms
license  : GPL-3.0+

Noise Transforms
"""

import tensorflow as tf

# Types
from typing import Optional, Tuple

# Local
# tf.debugging.set_log_device_placement(True)
from tfda.augmentations.noise_augmentations import (
    augment_gaussian_blur,
    augment_gaussian_noise
)
from tfda.base import DTFT, TFDABase
from tfda.defs import TFbF, TFbT, TFDAData


class GaussianNoiseTransform(TFDABase):
    """Adds additive Gaussian Noise.

    :param noise_variance:
             variance is uniformly sampled from that range
    :param p_per_sample:
    :param p_per_channel:
    :param per_channel:
             if True, each channel will get its own variance
             sampled from noise_variance
    :param data_key:

    CAREFUL: This transform will modify the value range of your data!
    """

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        return dataset.new_data(tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.defs.p_per_sample,
                lambda: augment_gaussian_noise(
                    x,
                    self.defs.noise_variance,
                    self.defs.p_per_channel,
                    self.defs.per_channel,
                ),
                lambda: x,
            ),
            dataset.data,
        ))


class GaussianBlurTransform(TFDABase):
    def __init__(
        self,
        blur_sigma: tf.Tensor = (1.0, 5.0),
        different_sigma_per_channel: tf.Tensor = True,
        different_sigma_per_axis: tf.Tensor = False,
            **kws,
    ) -> None:
        super().__init__(blur_sigma=blur_sigma, per_channel=different_sigma_per_channel, **kws)
        self.different_sigma_per_axis = different_sigma_per_axis

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        return dataset.new_data(tf.map_fn(
            lambda x: tf.cond(
                tf.less(tf.random.uniform(()), self.defs.p_per_sample),
                lambda: augment_gaussian_blur(
                    x,
                    self.defs.blur_sigma,
                    self.defs.per_channel,
                    self.defs.p_per_channel,
                ),
                lambda: x,
            ),
            dataset.data
        ))


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        dataset = next(
            iter(
                tf.data.Dataset.range(
                    8 * 1 * 40 * 56 * 40,
                    output_type=tf.float32,
                )
                .batch(40)
                .batch(56)
                .batch(40)
                .batch(2)
                .batch(4)
                .prefetch(4)
            )
        )
        data_dict = TFDAData(data=dataset)
        t = GaussianNoiseTransform(p_per_sample=1.0)

        tf.print(t(data_dict)["data"].shape)

        images = tf.random.uniform((8, 2, 20, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.float32
        )
        data_dict = TFDAData(data=images, seg=labels)
        tf.print(
            data_dict, data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)

        gbt = GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=TFbT,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )

        # Standard Library
        import time

        print(time.time())
        data_dict = gbt(TFDAData(seg=images, data=labels))
        tf.print(
            data_dict, data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)
        print(time.time())
        # data_dict = gbt(dict(data=images, seg=labels))
        # tf.print(
        #     data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        # )  # (8, 40, 376, 376) (8, 20, 376, 376)
        # tf.print(time.time())
