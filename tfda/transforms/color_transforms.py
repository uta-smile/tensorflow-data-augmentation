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
import tensorflow as tf

tf.debugging.set_log_device_placement(True)
# Others
from tfda.augmentations.color_augmentations import (
    augment_brightness_additive,
    augment_brightness_multiplicative,
    augment_contrast,
    augment_gamma,
)
from tfda.base import DTFT, TFT, TFDABase
from tfda.utils import TFbF, to_tf_bool, to_tf_float


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
        self,
        mu: float,
        sigma: float,
        per_channel: bool = True,
        **kws,
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


class GammaTransform(TFDABase):
    """
    Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

    :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
    larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
    Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
    smaller than 1 and the other half with > 1
    :param invert_image: whether to invert the image before applying gamma augmentation
    :param per_channel:
    :param data_key:
    :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
    the data will be transformed to match the mean and standard deviation before gamma augmentation
    :param p_per_sample:
    """

    def __init__(
        self,
        gamma_range: TFT = (0.5, 2),
        invert_image: TFT = TFbF,
        per_channel: TFT = TFbF,
        retain_stats: TFT = TFbF,
        **kws: TFT,
    ):
        super().__init__(**kws)
        self.retain_stats = retain_stats
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_gamma(
                    x,
                    self.gamma_range,
                    self.invert_image,
                    per_channel=self.per_channel,
                    retain_stats=self.retain_stats,
                ),
                lambda: x,
            ),
            data_dict[self.data_key],
        )
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

    # Others
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

        images = tf.random.uniform((8, 2, 20, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32
        )
        data_dict = {"data": images, "seg": labels}
        print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
        data_dict = GammaTransform(
            (0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1
        )(**data_dict)
        print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)

        tf.print(ts)
