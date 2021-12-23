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
date     : Dec  9, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : channel_selection_transforms.py
project  : transforms
license  : GPL-3.0+

Channel Selection Transforms
"""
# Standard Library
from dataclasses import dataclass
from warnings import warn

from tfda.base import DTFT, TFT, TFDABase
from tfda.utils import to_tf_bool, to_tf_float, to_tf_int

import tensorflow as tf


@dataclass
class DataChannelSelectionTransform(TFDABase):
    """Selects color channels from the batch and discards the others."""

    channels: TFT
    data_key: str = "data"

    def __hash__(self) -> int:
        return 1

    @tf.function
    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""

        data = tf.map_fn(
            lambda i: data_dict[self.data_key][:, to_tf_int(i)], self.channels
        )

        shape = data.shape

        data_dict[self.data_key] = tf.reshape(
            data, (shape[1], shape[0], *shape[2:])
        )

        # data_dict[self.data_key] = data_dict[self.data_key].map(
        #     lambda x: tf.stack(
        #         list(map(lambda i: x[:, i], self.channels)), axis=1
        #     )
        # )
        return data_dict


@dataclass
class SegChannelSelectionTransform(TFDABase):
    """Segmentations may have more than one channel.

    This transform selects segmentation channels.
    """

    channels: TFT
    label_key: str = "seg"
    keep_discarded: bool = False

    def __hash__(self) -> int:
        return 2

    @tf.function
    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        seg = data_dict.get(self.label_key)

        if to_tf_bool(seg is None):
            warn(
                "You used SegChannelSelectionTransform but "
                "there is no 'seg' key in your data_dict, returning "
                "data_dict unmodified",
                Warning,
            )
        else:
            # TODO: keep_discarded
            # if self.keep_discarded:
            shape = seg.shape[2:]
            seg = tf.map_fn(
                lambda i: seg[:, to_tf_int(i)],
                self.channels,
            )
            seg = tf.reshape(seg, (seg.shape[1], seg.shape[0], *shape))

            data_dict[self.label_key] = seg
            # data_dict[self.label_key] = seg.map(
            #     lambda x: tf.stack(
            #         list(map(lambda i: x[:, i], self.channels)), axis=1
            #     )
            # )
        return data_dict


if __name__ == "__main__":
    dataset = next(
        iter(
            tf.data.Dataset.range(
                8 * 1 * 1 * 40 * 56 * 40, output_type=tf.float32
            )
            .batch(40)
            .batch(56)
            .batch(40)
            .batch(1)
            .batch(2)
            .batch(4)
        )
    )

    with tf.device("/CPU:0"):

        dcst = DataChannelSelectionTransform(to_tf_float([]))
        tf.print(dcst(data=dataset)["data"].shape)

        scst = SegChannelSelectionTransform(tf.cast([0, 1], tf.float32))
        tf.print(scst(seg=dataset)["seg"].shape)
