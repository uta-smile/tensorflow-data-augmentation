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
filename : noise_transforms.py
project  : transforms
license  : GPL-3.0+

Noise Transforms
"""
from tfda.augmentations.noise_augmentations import augment_gaussian_noise
from tfda.base import DTFT, TFT, TFDABase
from tfda.utils import to_tf_bool, to_tf_float

import tensorflow as tf


class GaussianNoiseTransform(TFDABase):
    """Adds additive Gaussian Noise.

    :param noise_variance:
             variance is uniformly sampled from that range
    :param p_per_sample:
    :param p_per_channel:
    :param per_channel:
             if True, each channel will get its own variance
             sampled from noise_variance
    :param data_key:

    CAREFUL: This transform will modify the value range of your data!
    """

    # noise_variance: tuple[float, float] = 0, 0.1
    # p_per_sample: float = 1
    # p_per_channel: float = 1
    # per_channel: bool = False
    # data_key: str = "data"

    def call(self, **data_dict: TFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda x: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_gaussian_noise(
                    x,
                    self.noise_variance,
                    self.p_per_channel,
                    self.per_channel,
                ),
                lambda: x,
            ),
            data_dict[self.data_key],
        )
        # data_dict[self.data_key] = data_dict[self.data_key].map(
        #     lambda x_: tf.map_fn(
        #         lambda x: tf.cond(
        #             tf.random.uniform(()) < self.p_per_sample,
        #             lambda: augment_gaussian_noise(
        #                 x,
        #                 to_tf_float(self.noise_variance),
        #                 to_tf_float(self.p_per_channel),
        #                 to_tf_bool(self.per_channel),
        #             ),
        #             lambda: x,
        #         ),
        #         x_,
        #     ),
        # )
        return data_dict

    
class GaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma=(1, 5), data_key="data", label_key="seg", different_sigma_per_channel=True,
                 p_per_channel=1, p_per_sample=1):
        """

        :param blur_sigma:
        :param data_key:
        :param label_key:
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.label_key = label_key
        self.blur_sigma = blur_sigma

    @tf.function
    def __call__(self, **data_dict):
        data_list = []
        for b in range(len(data_dict[self.data_key])):
            if tf.random.uniform(()) < self.p_per_sample:
                data_b = augment_gaussian_blur(data_dict[self.data_key][b], self.blur_sigma,
                                               self.different_sigma_per_channel, self.p_per_channel)
            data_list.append(data_b)
        data_dict[self.data_key] = tf.stack(data_list)
        return data_dict


@tf.function
def augment_gaussian_blur(data_sample, sigma_range, per_channel=True, p_per_channel=1):
    if not per_channel:
        sigma = get_range_val(sigma_range)
    channel_list = []
    for c in range(data_sample.shape[0]):
        data_sample_channel = data_sample[c]
        if tf.random.uniform(()) <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range)
            # ToDo
            data_sample_channel = gaussian_filter(data_sample[c], sigma, order=0)
        channel_list.append(data_sample_channel)
    data_sample = tf.stack(channel_list)
    return data_sample


@tf.function
def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                if rnd_type == "uniform":
                    n_val = tf.random.uniform((), minval=value[0], maxval=value[1], dtype=tf.float32)
                elif rnd_type == "normal":
                    n_val = tf.random.normal((), mean=value[0], stddev=value[1], dtype=tf.float32)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single value or a list/tuple of len 2")
        return n_val
    else:
        return value 
    

if __name__ == "__main__":
    with tf.device("/CPU:0"):
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
        t = GaussianNoiseTransform(p_per_sample=0.3)

        tf.print(t(data=dataset)["data"].shape)
        
        
    images = tf.random.uniform((8, 2, 20, 376, 376))
    labels = tf.random.uniform((8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32)
    data_dict = {'data': images, 'seg': labels}
    print(data_dict.keys(), data_dict['data'].shape, data_dict['seg'].shape)  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
    data_dict = GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                      p_per_channel=0.5)(**data_dict)
    print(data_dict.keys(), data_dict['data'].shape, data_dict['seg'].shape)  # (8, 40, 376, 376) (8, 20, 376, 376)

