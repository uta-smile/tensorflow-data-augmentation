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
filename : utils.py
project  : tfda
license  : GPL-3.0+

Utils
"""
import tensorflow as tf

# Types
from typing import Any

TFT = tf.Tensor


TFf1 = tf.cast(1, tf.float32)
TFbF = tf.cast(False, tf.bool)


@tf.function
def to_tf_bool(x: Any) -> TFT:
    """Convert python bool to tf bool."""
    return tf.cast(x, tf.bool)


@tf.function
def to_tf_float(x: Any) -> TFT:
    """Convert python bool to tf float32."""
    return tf.cast(x, tf.float32)
