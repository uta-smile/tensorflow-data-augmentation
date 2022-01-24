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
date     : Jan 20, 2022
email    : Nasy <nasyxx+python@gmail.com>
           WlZhong <wxz9204@mavs.uta.edu>
filename : pyramid_augmentations.py
project  : transforms
license  : GPL-3.0+

https://github.com/MIC-DKFZ/nnUNet/
  nnunet/training/data_augmentation/pyramid_augmentations.py
"""

from numpy import float32
import tensorflow as tf

from tfda.base import TFDABase
from tfda.defs import TFDAData, TFbT
from tfda.augmentations.pyramid_augmentations import augment_apply_random_binary_operation, augment_move_seg_as_onehot_data_batch, do_remove_from_origin


class ApplyRandomBinaryOperatorTransform(TFDABase):
  def __init__(self, channel_idx, p_per_sample=0.3, strel_size=(1, 10), p_per_label=1, **kws) -> None:
      super().__init__(p_per_sample=p_per_sample, **kws)
      self._p_per_label = p_per_label
      self._channel_idx = channel_idx
      self._strel_size = strel_size

  @tf.function
  def call(self, data_dict: TFDAData) -> TFDAData:
    data = data_dict.data
    new_data = tf.map_fn(
      lambda b: augment_apply_random_binary_operation(data[b], self._channel_idx, self.defs.p_per_sample, self._strel_size, self._p_per_label),
      elems=tf.range(tf.shape(data)[0]),
      dtype=tf.float32
    )
    return data_dict.new_data(new_data)


class MoveSegAsOneHotToData(TFDABase):
  def __init__(self, channel_idx, all_seg_labels, remove_from_origin=TFbT, **kws):
    super.__init__(**kws)
    self._channel_idx = channel_idx
    self._all_seg_labels = all_seg_labels
    self._remove_from_origin = remove_from_origin

  @tf.function
  def call(self, data_dict: TFDAData) -> TFDAData:
    origin = data_dict.seg
    target = data_dict.data
    seg = origin[:, self._channel_idx: self._channel_idx+1]
    seg_onehot = tf.map_fn(
      lambda b: augment_move_seg_as_onehot_data_batch(seg[b][0], self._all_seg_labels),
      elems=tf.range(tf.shape(seg)[0]),
      dtype=tf.float32
    )

    target = tf.concat([target, seg_onehot], axis=1)
    origin = tf.cond(self._remove_from_origin, lambda: do_remove_from_origin(origin, self._channel_idx), lambda: origin)
    return TFDAData(data=target, seg=origin)
