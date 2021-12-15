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

from tfda.base import DTFD, TFD, TFDABase
from warnings import warn
import tensorflow as tf


@dataclass
class DataChannelSelectionTransform(TFDABase):
    """Selects color channels from the batch and discards the others.

    Args:
        channels (list of int): List of channels to be kept.
    """

    channels: list[int]
    data_key: str = "data"

    def call(self, **data_dict: TFD) -> DTFD:
        """Call the transform."""
        data_dict[self.data_key] = data_dict[self.data_key].map(
            lambda x: tf.stack(
                list(map(lambda i: x[:, i], self.channels)), axis=1
            )
        )
        return data_dict

    def __hash__(self) -> int:
        return 1


@dataclass
class SegChannelSelectionTransform(TFDABase):
    """Segmentations may have more than one channel.

    This transform selects segmentation channels.

    Args:
        channels (list of int): List of channels to be kept.
    """

    channels: list[int]
    label_key: str = "seg"
    keep_discarded: bool = False

    def __hash__(self) -> int:
        return 2

    def call(self, **data_dict: TFD) -> DTFD:
        """Call the transform."""
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn(
                "You used SegChannelSelectionTransform but "
                "there is no 'seg' key in your data_dict, returning "
                "data_dict unmodified",
                Warning,
            )
        else:
            # TODO: keep_discarded
            # if self.keep_discarded:
            data_dict[self.label_key] = seg.map(
                lambda x: tf.stack(
                    list(map(lambda i: x[:, i], self.channels)), axis=1
                )
            )
        return data_dict


if __name__ == "__main__":
    dataset = tf.data.Dataset.range(10).batch(5).batch(2)

    dcst = DataChannelSelectionTransform([2, 3, 1])
    assert tf.math.reduce_all(
        next(iter(dcst(data=dataset)["data"]))
        == tf.cast([[2, 3, 1], [7, 8, 6]], dtype=tf.int64)
    )

    scst = SegChannelSelectionTransform([0, 2])
    assert tf.math.reduce_all(
        next(iter(scst(seg=dataset)["seg"]))
        == tf.cast([[0, 2], [5, 7]], dtype=tf.int64)
    )
