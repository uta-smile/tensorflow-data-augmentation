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

# Others
from tfda.augmentations.noise_augmentations import (
    augment_gaussian_blur,
    augment_gaussian_noise,
)
from tfda.base import DTFT, TFT, TFDABase
from tfda.utils import TFbF, TFbT


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

    # noise_variance: tuple[float, float] = 0, 0.1
    # p_per_sample: float = 1
    # p_per_channel: float = 1
    # per_channel: bool = False
    # data_key: str = "data"

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_gaussian_noise(
                    x,
                    self.noise_variance,
                    self.p_per_channel,
                    self.per_channel,
                ),
                lambda: x,
            ),
            data_dict[self.data_key],
        )
        # data_dict[self.data_key] = data_dict[self.data_key].map(
        #     lambda x_: tf.map_fn(
        #         lambda x: tf.cond(
        #             tf.random.uniform(()) < self.p_per_sample,
        #             lambda: augment_gaussian_noise(
        #                 x,
        #                 to_tf_float(self.noise_variance),
        #                 to_tf_float(self.p_per_channel),
        #                 to_tf_bool(self.per_channel),
        #             ),
        #             lambda: x,
        #         ),
        #         x_,
        #     ),
        # )
        return data_dict


class GaussianBlurTransform(TFDABase):
    def __init__(
        self,
        blur_sigma: TFT = (1.0, 5.0),
        different_sigma_per_channel: TFT = TFbT,
        different_sigma_per_axis: TFT = TFbF,
        **kws: TFT
    ) -> None:
        super().__init__(**kws)
        self.different_sigma_per_channel = different_sigma_per_channel
        self.blur_sigma = blur_sigma

    def call(self, **data_dict: TFT) -> DTFT:
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.less(tf.random.uniform(()), self.p_per_sample),
                lambda: augment_gaussian_blur(
                    x,
                    self.blur_sigma,
                    self.different_sigma_per_channel,
                    self.p_per_channel,
                ),
                lambda: x,
            ),
            data_dict[self.data_key],
        )
        return data_dict



if __name__ == "__main__":
    with tf.device("/CPU:0"):
        dataset = next(
            iter(
                tf.data.Dataset.range(
                    8 * 1 * 1 * 40 * 56 * 40, output_type=tf.float32
                )
                .batch(40)
                .batch(56)
                .batch(40)
                .batch(1)
                .batch(2)
                .batch(4)
            )
        )
        t = GaussianNoiseTransform(p_per_sample=0.3)

        tf.print(t(data=dataset)["data"].shape)

        images = tf.random.uniform((8, 2, 20, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.float32
        )
        data_dict = {"data": images, "seg": labels}
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
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
        data_dict = gbt(seg=images, data=labels)
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)
        print(time.time())
        data_dict = gbt(data=images, seg=labels)
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)
        print(time.time())
