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
import abc
from itertools import chain
from tfda.utils import to_tf_float

import tensorflow as tf

from tfda.defs import DTFT, Seqs

# tf.debugging.set_log_device_placement(True)


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
        noise_variance: tf.Tensor = (0., 0.1),
        different_sigma_per_channel: tf.Tensor = True,
        **kws,
    ) -> None:
        super().__init__(**kws)
        self.p_per_sample = tf.convert_to_tensor(p_per_sample)
        self.p_per_channel = tf.convert_to_tensor(p_per_channel)
        self.per_channel = tf.convert_to_tensor(per_channel)
        self.contrast_range = tf.convert_to_tensor(contrast_range)
        self.multiplier_range = tf.convert_to_tensor(multiplier_range)
        self.preserve_range = tf.convert_to_tensor(preserve_range)
        self.noise_variance = tf.convert_to_tensor(noise_variance)
        self.different_sigma_per_channel = tf.convert_to_tensor(different_sigma_per_channel)

        self.data_key = data_key
        self.label_key = label_key


class RndTransform(TFDABase):
    """Random transform."""

    def __init__(self, transform: TFDABase, prob: float = 0.5, **kws):
        super().__init__(**kws)
        self.transform = transform
        self.prob = prob

    @tf.function
    def call(self, **data_dict: tf.Tensor) -> DTFT:
        """Call the Rnd transform."""
        return (
            tf.random.uniform() < self.prob
            and self.transform(data_dict)
            or data_dict
        )


class IDTransform(TFDABase):
    """Identity transform."""

    @tf.function
    def call(self, **data_dict: tf.Tensor) -> DTFT:
        """Call the transform."""
        return data_dict


class Compose(TFDABase):
    """Compose transforms."""

    def __init__(self, transforms: Seqs[TFDABase], **kws) -> None:
        super().__init__(**kws)
        self.transforms = transforms

    def add(self, transform: TFDABase) -> "Compose":
        """Add transform."""
        self.transforms = chain(self.transforms, (transform,))
        return self

    def call(self, data_dict: DTFT) -> DTFT:
        """Call the transforms."""
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict

    def __repr__(self) -> str:
        return f"{type(self).__name__} ( {repr(self.transforms)} )"


if __name__ == "__main__":

    class _Add1Transform(TFDABase):
        """Add 1 transform.

        For test only
        """

        @tf.function
        def add1(self, x: tf.Tensor) -> tf.Tensor:
            """Add 1."""
            return x + 1

        @tf.function
        def call(self, **data_dict: tf.Tensor) -> DTFT:
            """Call the add 1 transform."""

            for key, data in data_dict.items():
                data_dict[key] = self.add1(data)

            return data_dict

    data_sample = next(
        iter(
            tf.data.Dataset.range(20, output_type=tf.float32).batch(5).batch(2)
        )
    )

    tf.print(_Add1Transform()(x=data_sample))
