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
date     : Dec 14, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : color_transforms.py
project  : transforms
license  : GPL-3.0+

Color Transforms
"""
from tfda.augmentations.color_augmentations import (
    augment_brightness_additive,
    augment_brightness_multiplicative,
    augment_contrast
)
from tfda.base import DTFT, TFT, TFDABase
from tfda.utils import to_tf_bool, to_tf_float

import tensorflow as tf


class ColorTrans(TFDABase):
    """Base of color transform."""

    def __init__(self, **data_dict):
        data_dict["per_channel"] = True
        super().__init__(**data_dict)


class ContrastAugmentationTransform(ColorTrans):
    """Contrast augmentataion transform."""

    # contrast_range: tuple[float, float] = (0.75, 1.25)
    # preserve_range: bool = True
    # per_channel: bool = True
    # data_key: str = "data"
    # p_per_sample: float = 1
    # p_per_channel: float = 1

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_contrast(
                    x,
                    self.contrast_range,
                    self.preserve_range,
                    self.per_channel,
                    self.p_per_channel,
                ),
                lambda: x,
            ),
            data_dict[self.data_key],
        )

        # data_dict[self.data_key] = data_dict[self.data_key].map(
        #     lambda x_: tf.map_fn(
        #         lambda x: tf.cond(
        #             tf.random.uniform(()) < self.p_per_sample,
        #             lambda: augment_contrast(
        #                 x,
        #                 to_tf_float(self.contrast_range),
        #                 to_tf_bool(self.preserve_range),
        #                 to_tf_bool(self.per_channel),
        #                 to_tf_float(self.p_per_channel),
        #             ),
        #             lambda: x,
        #         ),
        #         x_,
        #     ),
        # )
        return data_dict


class BrightnessTransform(ColorTrans):
    """Augments the brightness of data."""

    # mu: float
    # sigma: float
    # per_channel: bool = True
    # data_key: str = "data"
    # p_per_sample: float = 1
    # p_per_channel: float = 1

    def __init__(
        self, mu: float, sigma: float, per_channel: bool = True, **kws
    ) -> None:
        super().__init__(**kws)
        self.mu = to_tf_float(mu)
        self.sigma = to_tf_float(sigma)
        self.per_channel = to_tf_bool(per_channel)

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_brightness_additive(
                    x,
                    self.mu,
                    self.sigma,
                    self.per_channel,
                    self.p_per_channel,
                ),
                lambda: x,
            ),
            data_dict[self.data_key],
        )
        # data_dict[self.data_key] = data_dict[self.data_key].map(
        #     lambda x_: tf.map_fn(
        #         lambda x: tf.cond(
        #             tf.random.uniform(()) < self.p_per_sample,
        #             lambda: augment_brightness_additive(
        #                 x,
        #                 to_tf_float(self.mu),
        #                 to_tf_float(self.sigma),
        #                 to_tf_bool(self.per_channel),
        #                 to_tf_float(self.p_per_channel),
        #             ),
        #             lambda: x,
        #         ),
        #         x_,
        #     ),
        # )
        return data_dict


class BrightnessMultiplicativeTransform(ColorTrans):
    """Augments the brightness of data."""

    # multiplier_range: tuple[float, float] = (0.5, 2)
    # per_channel: bool = True
    # data_key: str = "data"
    # p_per_sample: float = 1

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_brightness_multiplicative(
                    x,
                    self.multiplier_range,
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
        #             lambda: augment_brightness_multiplicative(
        #                 x,
        #                 to_tf_float(self.multiplier_range),
        #                 to_tf_bool(self.per_channel),
        #             ),
        #             lambda: x,
        #         ),
        #         x_,
        #     ),
        # )
        return data_dict


if __name__ == "__main__":
    dataset = (
        tf.data.Dataset.range(8 * 1 * 1 * 40 * 56 * 40, output_type=tf.float32)
        .batch(40)
        .batch(56)
        .batch(40)
        .batch(1)
        .batch(2)
        .batch(4)
    )

    data_sample = next(iter(dataset))

    from tfda.base import Compose

    ts = Compose(
        [
            ContrastAugmentationTransform(),
            BrightnessTransform(0, 0.1),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.2
            ),
        ]
    )

    with tf.device("/CPU:0"):
        tf.print(ts(data=data_sample)["data"].shape)
