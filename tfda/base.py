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

import tensorflow as tf

# Types
from typing import Any, Sequence, TypeVar, Union

DAny = dict[str, Any]
T = TypeVar("T")
Seqs = Union[Sequence[T], chain[T]]


class TFDABase:
    """Tensorflow data augmentation base."""

    @abc.abstractmethod
    def call(self, **kws: Any):
        """Call the base transform."""
        raise NotImplementedError("Abstract, so implement")

    def __call__(self, **kws: Any) -> DAny:
        """Call call function."""
        return self.call(**kws)


@dataclass
class RndTransform(TFDABase):
    """Random transform."""

    transform: TFDABase
    prob: float = 0.5

    def call(self, **kws: Any) -> DAny:
        """Call the Rnd transform."""
        return tf.random.uniform() < self.prob and self.transform(**kws) or kws


class IDTransform(TFDABase):
    """Identity transform."""

    def call(self, **kws: Any) -> DAny:
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

    def call(self, **kws: Any) -> DAny:
        """Call the transforms."""
        for transform in self.transforms:
            kws = transform(**kws)
        return kws

    def __repr__(self) -> str:
        return f"{type(self).__name__} ( {repr(self.transforms)} )"


if __name__ == "__main__":

    class _Add1Transform(TFDABase):
        """Add 1 transform.

        For test only
        """

        def call(self, **kws: int) -> dict[str, int]:
            """Call the add 1 transform."""
            for k, v in kws.items():
                kws[k] = v + 1
            return kws

    assert Compose([_Add1Transform(), _Add1Transform()])(x=1) == {"x": 3}
