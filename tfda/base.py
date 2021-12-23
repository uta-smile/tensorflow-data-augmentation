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
# from __future__ import annotations

# Standard Library
import abc
from itertools import chain
from tfda.utils import to_tf_float, to_tf_bool

import tensorflow as tf

# Types
from typing import Any, Dict, Iterable, Sequence, TypeVar, Union

TFT = tf.Tensor
DTFT = Dict[str, tf.Tensor]
T = TypeVar("T")
Seqs = Union[Sequence[T], Iterable[T]]


# tf.debugging.set_log_device_placement(True)


class TFDABase:
    """Tensorflow data augmentation base."""

    def __init__(
        self,
        data_key: TFT = "data",
        label_key: TFT = "seg",
        p_per_sample: TFT = 1,
        p_per_channel: TFT = 1,
        per_channel: TFT = False,
        contrast_range: TFT = (0.75, 1.25),
        multiplier_range: TFT = (0.5, 2),
        preserve_range: TFT = True,
        noise_variance: TFT = (0, 0.1),
    ) -> None:
        self.p_per_sample = to_tf_float(p_per_sample)
        self.p_per_channel = to_tf_float(p_per_channel)
        self.per_channel = to_tf_bool(per_channel)
        self.contrast_range = to_tf_float(contrast_range)
        self.multiplier_range = to_tf_float(multiplier_range)
        self.preserve_range = to_tf_bool(preserve_range)
        self.noise_variance = to_tf_float(noise_variance)

        self.data_key = data_key

    @abc.abstractmethod
    def call(self, **data_dict: TFT):
        """Call the base transform."""
        raise NotImplementedError("Abstract, so implement")

    @tf.function
    def __call__(self, *args, **kws):
        return self.call(*args, **kws)


class RndTransform(TFDABase):
    """Random transform."""

    def __init__(self, transform: TFDABase, prob: float = 0.5, **kws):
        super().__init__(**kws)
        self.transform = transform
        self.prob = prob

    @tf.function
    def call(self, **data_dict: TFT) -> DTFT:
        """Call the Rnd transform."""
        return (
            tf.random.uniform() < self.prob
            and self.transform(**data_dict)
            or data_dict
        )


class IDTransform(TFDABase):
    """Identity transform."""

    @tf.function
    def call(self, **data_dict: TFT) -> DTFT:
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

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transforms."""
        for transform in self.transforms:
            data_dict = transform(**data_dict)
        return data_dict

    def __hash__(self):
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__} ( {repr(self.transforms)} )"


if __name__ == "__main__":

    class _Add1Transform(TFDABase):
        """Add 1 transform.

        For test only
        """

        @tf.function
        def add1(self, x: TFT) -> TFT:
            """Add 1."""
            return x + 1

        @tf.function
        def call(self, **data_dict: TFT) -> DTFT:
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
