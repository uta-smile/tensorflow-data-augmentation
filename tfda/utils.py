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
# Standard Library
import math as m

import tensorflow as tf

# Types
from typing import Any

TFf0 = tf.cast(0, tf.float32)
TFf1 = tf.cast(1, tf.float32)
TFi1 = tf.cast(1, tf.int64)
TFbF = tf.cast(False, tf.bool)
TFbT = tf.cast(True, tf.bool)
pi = tf.constant(m.pi)
nan = tf.constant(m.nan)


@tf.function
def to_tf_bool(x: Any) -> tf.Tensor:
    """Convert python bool to tf bool."""
    return tf.cast(x, tf.bool)


@tf.function
def to_tf_float(x: Any) -> tf.Tensor:
    """Convert python bool to tf float32."""
    return tf.cast(x, tf.float32)


@tf.function
def to_tf_int(x: Any) -> tf.Tensor:
    """Convert python bool to tf int64."""
    return tf.cast(x, tf.int64)


@tf.function(experimental_follow_type_hints=True)
def isnan(x: tf.Tensor) -> tf.Tensor:
    """Check if there is any nan in it."""
    return tf.math.reduce_any(tf.math.is_nan(x))
