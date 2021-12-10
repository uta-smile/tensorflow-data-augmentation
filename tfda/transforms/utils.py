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
filename : utils.py
project  : transforms

Utils transforms
"""
# Standard Library
from dataclasses import dataclass

from tfda.base import DTFD, TFD, Compose, TFDABase

import tensorflow as tf

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

        def call(self, **kws: TFD) -> DTFD:
            """Call the add 1 transform."""
            for k, v in kws.items():
                kws[k] = v.map(self.add1)
            return kws

    assert list(
        Compose([_Add1Transform(), _Add1Transform()])(
            x=tf.data.Dataset.range(3)
        )["x"].as_numpy_iterator()
    ) == list(tf.data.Dataset.range(2, 5).as_numpy_iterator())
