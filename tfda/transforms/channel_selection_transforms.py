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
import tensorflow as tf

# Local
from tfda.base import DTFT, TFDABase
from tfda.defs import TFbT, nan
from tfda.utils import to_tf_int


class DataChannelSelectionTransform(TFDABase):
    """Selects color channels from the batch and discards the others."""

    def __init__(self, channels: tf.Tensor, **kws: tf.Tensor) -> None:
        super().__init__(**kws)
        self.channels = channels

    def call(self, data_dict: DTFT) -> DTFT:
        """Call the transform."""

        data = tf.map_fn(
            lambda i: data_dict[self.data_key][:, to_tf_int(i)], self.channels
        )

        shape = tf.shape(data)
        tf.print(shape)

        data_dict[self.data_key] = tf.reshape(
            data, (shape[1], shape[0], *shape[2:])
        )
        return data_dict


class SegChannelSelectionTransform(TFDABase):
    """Segmentations may have more than one channel.

    This transform selects segmentation channels.
    """

    def __init__(
        self,
        channels: tf.Tensor,
        keep_discarded: tf.Tensor = TFbT,
        **kws: tf.Tensor
    ) -> None:
        super().__init__(**kws)
        self.channels = channels
        self.keep_discarded = keep_discarded

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: DTFT) -> DTFT:
        """Call the transform."""
        data_dict = data_dict.copy()
        seg = data_dict.get(self.label_key, nan)

        if tf.math.reduce_any(tf.math.is_nan(seg)):
            tf.get_logger().warn(
                "You used SegChannelSelectionTransform but "
                "there is no 'seg' key in your data_dict, returning "
                "data_dict unmodified",
            )
        else:
            # TODO: keep_discarded
            # if self.keep_discarded:
            shape = tf.shape(seg)[2:]
            seg = tf.map_fn(
                lambda i: seg[:, to_tf_int(i)],
                self.channels,
            )
            seg = tf.reshape(seg, (tf.shape(seg)[1], tf.shape(seg)[0], *shape))

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
            .prefetch(4)
        )
    )

    with tf.device("/CPU:0"):

        # dcst = DataChannelSelectionTransform(to_tf_float([]))
        # tf.print(dcst(dict(data=dataset))["data"].shape)

        scst = SegChannelSelectionTransform(tf.cast([0, 1], tf.float32))
        tf.print(scst(dict(seg=dataset))["seg"].shape)
