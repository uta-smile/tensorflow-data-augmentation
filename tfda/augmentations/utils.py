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
date     : Dec 16, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : augmentations
license  : GPL-3.0+

Augmentation Utils
"""

import tensorflow as tf
from tfda.base import TFT
from tfda.utils import to_tf_float, to_tf_int, TFbF, TFbT


def create_zero_centered_coordinate_mesh(shape: TFT) -> TFT:
    tmp = tf.map_fn(
        lambda x: tf.range(x, dtype=tf.float32),
        shape,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=[None], dtype=tf.float32
        ),
    )
    coords = tf.cast(tf.meshgrid(*tmp, indexing="ij"), dtype=tf.float32)

    coords = tf.map_fn(
        lambda i: coords[to_tf_int(i)]
        - ((to_tf_float(shape) - 1) / 2)[to_tf_int(i)],
        tf.range(coords.shape[0], dtype=tf.float32),
    )
    return coords





if __name__ == '__main__':
    patch_size = tf.constant([40, 56, 40])
