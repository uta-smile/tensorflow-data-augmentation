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
import tensorflow as tf

from tfda.base import TFDABase
from tfda.defs import TFDAData, TFbT
from tfda.augmentations.pyramid_augmentations import augment_apply_random_binary_operation, augment_move_seg_as_onehot_data_batch, do_remove_from_origin


class RemoveRandomConnectedComponentFromOneHotEncodingTransform(TFDABase):

    def __init__(
        self,
        channel_idx,
        key="data",
        p_per_sample=0.2,
        fill_with_other_class_p=0.25,
        dont_do_if_covers_more_than_X_percent=0.25,
        p_per_label=1,
        **kws,
    ) -> None:
        super().__init__(p_per_sample=p_per_sample, **kws)
        self.channel_idx = channel_idx
        self.key = key
        self.fill_with_other_class_p = fill_with_other_class_p
        self.dont_do_if_covers_more_than_X_percent = dont_do_if_covers_more_than_X_percent
        self.p_per_label = p_per_label

    @tf.function
    def call(self, data_dict: TFDAData) -> TFDAData:
      data = data_dict.data
      return data_dict.new_data(tf.map_fn(lambda b: self.map1(data, b), tf.shape(data)[0]))


    @tf.function
    def map1(self, data, b) -> tf.Tensor:
      return tf.cond(tf.less(tf.random.uniform(()), self.defs.p_per_sample),
                 lambda: tf.map_fn(lambda c: self.map2(data, b, c), tf.shape(data)[1]),
                 lambda: data[b])

    @tf.function
    def map2(self, data, b, c) -> tf.Tensor:
      return tf.cond(tf.less(tf.random.uniform(()), self.p_per_label),
                 lambda: self.cond1(data, b, c),
                 lambda: data[b, c])

    @tf.funcion
    def cond1(self, data, b, c) -> tf.Tensor:
      workon = tf.identity(data[b, c])
      num_voxels = tf.math.reduce_prod(tf.shape(workon))
      # TODO: lable
      lab, num_comp = label(workon, return_num = True)
      return tf.cond(tf.greater(num_comp, 0), lambda: self.cond2(data, b, c, lab, num_comp, num_voxels), lambda: data[b, c])


    @tf.function
    def cond2(self, data, b, c, lab, num_comp, num_voxels) -> tf.Tensor:
      component_sizes = tf.map_fn(lambda i: tf.reduce_sum(tf.cast(tf.equal(lab, i), tf.int64)), tf.range(1, num_comp + 1))
      component_ids_w0 = tf.map_fn(lambda i: tf.cond(tf.less(component_sizes[i], num_voxels * self.dont_do_if_covers_more_than_X_percent),
                                             lambda: i + 1, lambda: 0), tf.range(num_comp))
      length = tf.reduce_sum(tf.cast(tf.not_equal(component_ids_w0, 0), tf.int64))
      comps = tf.TensorArray(tf.int64, size=length)
      idx = 0
      for com in component_ids_w0:
          if com > 0:
              comps = comps.write(idx, com)
              idx = idx + 1
      component_ids = comps.stack()
      return tf.cond(tf.greater(length, 0), lambda: tf.cond3(data, b, c, lab, component_ids), lambda: data[b, c])

    @tf.function
    def cond3(self, data, b, c, lab, component_ids) -> tf.Tensor:
        idx = tf.random.normal((), 0, tf.size(component_ids), tf.int32)
        databc = tf.where(tf.equal(lab, component_ids[idx]), 0, data[b, c])
        other_ch = tf.where(tf.not_equal(self.channel_idx, c))
        oidx = tf.random.normal((), 0, tf.shape(other_ch)[0], tf.int32)

        databo = tf.cond(tf.greater(tf.size(other_ch), 0), lambda: tf.where(tf.equal(lab, component_ids[idx]), 1, data[b, other_ch[oidx]]))



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
