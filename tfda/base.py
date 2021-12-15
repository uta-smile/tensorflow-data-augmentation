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
from __future__ import annotations

# Standard Library
import abc
from dataclasses import dataclass
from itertools import chain
from functools import wraps

import tensorflow as tf

# Types
from typing import Any, Sequence, TypeVar, Union

TFD = tf.data.Dataset
DTFD = dict[str, tf.data.Dataset]
T = TypeVar("T")
Seqs = Union[Sequence[T], chain[T]]


class TFDABase:
    """Tensorflow data augmentation base."""

    @abc.abstractmethod
    def call(self, **kws: Any):
        """Call the base transform."""
        raise NotImplementedError("Abstract, so implement")

    @tf.function
    def __call__(self, **kws: Any) -> DTFD:
        """Call call function."""
        return self.call(**kws)


@dataclass(unsafe_hash=True)
class RndTransform(TFDABase):
    """Random transform."""

    transform: TFDABase
    prob: float = 0.5

    @tf.function
    def call(self, **kws: Any) -> DTFD:
        """Call the Rnd transform."""
        return tf.random.uniform() < self.prob and self.transform(**kws) or kws


class IDTransform(TFDABase):
    """Identity transform."""

    @tf.function
    def call(self, **kws: Any) -> DTFD:
        """Call the transform."""
        return kws


@dataclass
class Compose(TFDABase):
    """Compose transforms."""

    transforms: Seqs[TFDABase]

    def add(self, transform: TFDABase) -> Compose:
        """Add transform."""
        self.transforms = chain(self.transforms, (transform,))
        return self

    def call(self, **kws: Any) -> DTFD:
        """Call the transforms."""
        for transform in self.transforms:
            kws = transform(**kws)
        return kws

    def __hash__(self):
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__} ( {repr(self.transforms)} )"


if __name__ == "__main__":

    class _Add1Transform(TFDABase):
        """Add 1 transform.

        For test only
        """

        @staticmethod
        @tf.function
        def add1(x: TFD) -> TFD:
            """Add 1."""
            return x + 1

        @tf.function
        def call(self, **kws: TFD) -> DTFD:
            """Call the add 1 transform."""
            for k, v in kws.items():
                kws[k] = v.map(self.add1)
            return kws

    with tf.device("/CPU:0"):
        tf.print(
            list(
                _Add1Transform()(x=tf.data.Dataset.range(3))[
                    "x"
                ].as_numpy_iterator()
            )
        )

        assert list(
            Compose([_Add1Transform(), _Add1Transform()])(
                x=tf.data.Dataset.range(3)
            )["x"].as_numpy_iterator()
        ) == list(tf.data.Dataset.range(2, 5).as_numpy_iterator())
