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
date     : Dec 30, 2021
email    : Nasy <nasyxx+python@gmail.com>
filename : types.py
project  : tfda
license  : GPL-3.0+

TFDA defaults
"""

# Standard Library
import math as m
import os

import tensorflow as tf

# Types
from typing import Dict, Iterable, Optional, Sequence, TypeVar, Union

DTFT = Dict[str, tf.Tensor]
T = TypeVar("T")
Seqs = Union[Sequence[T], Iterable[T]]


TFbF = tf.cast(False, tf.bool)
TFbT = tf.cast(True, tf.bool)
pi = tf.constant(m.pi)
nan = tf.constant(m.nan)

DR = (-15.0 / 360 * 2.0 * pi, 15.0 / 360 * 2.0 * pi)


class TFDAData(tf.experimental.BatchableExtensionType):
    """TFDA data set."""

    shape: tf.TensorShape
    data: tf.Tensor
    seg: tf.Tensor

    def __init__(self, data: tf.Tensor = nan, seg: tf.Tensor = nan) -> None:
        self.data = tf.convert_to_tensor(data)
        self.seg = tf.convert_to_tensor(seg)
        self.shape = self.data.shape

    def new_data(self, data: tf.Tensor) -> "TFDAData":
        """Replace data with new data."""
        return TFDAData(data, self.seg)

    @tf.function(experimental_follow_type_hints=True)
    def __getitem__(self, key: str) -> tf.Tensor:
        """Get item of default params."""
        return self.__getattribute__(key)

    def __repr__(self) -> str:
        return (
            f"TFDAData(data.shape = {self.data.shape}, "
            f"seg.shape = {self.seg.shape})"
        )


class TFDADefs(tf.experimental.ExtensionType):
    """TFDA transform defaults params."""

    data_key: str = "data"
    label_key: str = "seg"
    p_per_sample: tf.Tensor = 1.0
    p_per_channel: tf.Tensor = 1.0
    per_channel: tf.Tensor = False
    contrast_range: tf.Tensor = (0.75, 1.25)
    multiplier_range: tf.Tensor = (0.5, 2.0)
    preserve_range: tf.Tensor = True
    noise_variance: tf.Tensor = (0.0, 0.1)
    different_sigma_per_channel: tf.Tensor = True
    gamma_range: tf.Tensor = (0.5, 2)
    invert_image: tf.Tensor = False
    retain_stats: tf.Tensor = False
    blur_sigma: tf.Tensor = (1., 5.)
    zoom_range: tf.Tensor = (0.5, 1.)
    order_downsample: tf.Tensor = 1
    order_upsample: tf.Tensor = 0
    ignore_axes: tf.Tensor = nan


class TFDADefault3DParams(tf.experimental.ExtensionType):
    """TFDA default 3D augmentation params."""

    patch_size_for_spatial_transform: tf.Tensor = tf.fill([3], nan)
    selected_data_channels: tf.Tensor = tf.reshape(nan, (-1, 1))
    selected_seg_channels: tf.Tensor = tf.reshape(nan, (-1, 1))
    do_elastic: tf.Tensor = True
    elastic_deform_alpha: tf.Tensor = (0.0, 900.0)
    elastic_deform_sigma: tf.Tensor = (9.0, 13.0)
    p_eldef: tf.Tensor = 0.2
    do_scaling: tf.Tensor = True
    scale_range: tf.Tensor = (0.85, 1.25)
    independent_scale_factor_for_each_axis: tf.Tensor = False
    p_independent_scale_per_axis: tf.Tensor = 1.0
    p_scale: tf.Tensor = 0.2
    do_rotation: tf.Tensor = True
    rotation_x: tf.Tensor = DR
    rotation_y: tf.Tensor = DR
    rotation_z: tf.Tensor = DR
    rotation_p_per_axis: tf.Tensor = 1.0
    p_rot: tf.Tensor = 0.2
    random_crop: tf.Tensor = False
    random_crop_dist_to_border: tf.Tensor = nan
    do_gamma: tf.Tensor = True
    gamma_retain_stats: tf.Tensor = True
    gamma_range: tf.Tensor = (0.7, 1.5)
    p_gamma: tf.Tensor = 0.3
    do_mirror: tf.Tensor = True
    mirror_axes: tf.Tensor = (0, 1, 2)
    dummy_2D: tf.Tensor = False
    mask_was_used_for_normalization: tf.Tensor = nan
    border_mode_data: tf.Tensor = "constant"
    all_segmentation_labels: tf.Tensor = nan  # used for cascade
    move_last_seg_chanel_to_data: tf.Tensor = False  # used for cascade
    cascade_do_cascade_augmentations: tf.Tensor = False  # used for cascade
    cascade_random_binary_transform_p: tf.Tensor = 0.4
    cascade_random_binary_transform_p_per_label: tf.Tensor = 1.0
    cascade_random_binary_transform_size: tf.Tensor = (1, 8)
    cascade_remove_conn_comp_p: tf.Tensor = 0.2
    cascade_remove_conn_comp_max_size_percent_threshold: tf.Tensor = 0.15
    cascade_remove_conn_comp_fill_with_other_class_p: tf.Tensor = 0.0
    do_additive_brightness: tf.Tensor = False
    additive_brightness_p_per_sample: tf.Tensor = 0.15
    additive_brightness_p_per_channel: tf.Tensor = 0.5
    additive_brightness_mu: tf.Tensor = 0.0
    additive_brightness_sigma: tf.Tensor = 0.1
    num_threads: tf.Tensor = int(os.getenv("nnUNet_n_proc_DA", 12))
    num_cached_per_thread: tf.Tensor = 1

    @tf.function(experimental_follow_type_hints=True)
    def __getitem__(self, key: str) -> tf.Tensor:
        """Get item of default params."""
        return self.__getattribute__(key)

    @tf.function(experimental_follow_type_hints=True)
    def get(self, key: str) -> tf.Tensor:
        """Get item of default params."""
        return self.__getattribute__(key)


class TFDADefault2DParams(TFDADefault3DParams):
    """TFDA default 2D augmentation params."""

    pass
