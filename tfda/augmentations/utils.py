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

# Tensorflow
import tensorflow as tf

# Local
from tfda.defs import nan
from tfda.utils import to_tf_float, to_tf_int

# tf.debugging.set_log_device_placement(True)


@tf.function(experimental_follow_type_hints=True)
def to_one_hot(seg: tf.Tensor, all_seg_labels: tf.Tensor = nan) -> tf.Tensor:
    all_seg_labels = tf.cast(all_seg_labels, tf.float32)

    return tf.map_fn(
        lambda s: tf.map_fn(
            lambda i: tf.where(tf.equal(seg[s], all_seg_labels[i]), 1.0, 0.0),
            tf.range(tf.size(all_seg_labels)),
            fn_output_signature=tf.float32,
        ),
        tf.range(tf.shape(seg)[0]),
        fn_output_signature=tf.float32,
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    ]
)
def get_range_val(value: tf.Tensor, rnd_type: tf.Tensor = "uniform"):
    # TODO: different rank values
    return tf.case(
        [
            (
                tf.logical_and(
                    tf.equal(tf.shape(value)[0], 2),
                    tf.equal(rnd_type, "uniform"),
                ),
                lambda: tf.random.uniform(
                    (), minval=value[0], maxval=value[1], dtype=tf.float32
                ),
            ),
            (
                tf.equal(rnd_type, "normal"),
                lambda: tf.random.normal(
                    (), mean=value[0], stddev=value[1], dtype=tf.float32
                ),
            ),
        ]
    )


@tf.function(experimental_follow_type_hints=True)
def create_zero_centered_coordinate_mesh(shape: tf.Tensor) -> tf.Tensor:
    tmp = tf.map_fn(
        lambda x: tf.range(x, dtype=tf.float32),
        shape,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=[None], dtype=tf.float32
        ),
    )

    # TODO: change hardcode to others
    # How to use *tmp in tensorflow graph?
    # tf.meshgrid(*tmp, indexing="ij")
    coords = tf.cast(
        tf.meshgrid(tmp[0], tmp[1], tmp[2], indexing="ij"), dtype=tf.float32
    )

    return tf.map_fn(
        lambda i: coords[to_tf_int(i)]
        - ((to_tf_float(shape) - 1) / 2)[to_tf_int(i)],
        tf.range(tf.shape(coords)[0], dtype=tf.float32),
    )


@tf.function(experimental_follow_type_hints=True)
def elastic_deform_coordinates(
    coordinates: tf.Tensor, alpha: tf.Tensor, sigma: tf.Tensor
):
    coordinates = tf.cast(coordinates, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    return (
        tf.map_fn(
            lambda _: gaussian_filter(
                (tf.random.uniform(tf.shape(coordinates)[1:]) * 2 - 1),
                sigma,
                mode=1,
            )
            * alpha,
            tf.range(tf.shape(coordinates)[0], dtype=tf.float32),
        )
        + coordinates
    )


# rotation relate
@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_x_3d(
    angle: tf.Tensor, matrix: tf.Tensor
) -> tf.Tensor:
    rotation_x = tf.cast(
        [
            [1, 0, 0],
            [0, tf.cos(angle), -tf.sin(angle)],
            [0, tf.sin(angle), tf.cos(angle)],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_y_3d(
    angle: tf.Tensor, matrix: tf.Tensor
) -> tf.Tensor:
    rotation_x = tf.cast(
        [
            [tf.cos(angle), 0, tf.sin(angle)],
            [0, 1, 0],
            [-tf.sin(angle), 0, tf.cos(angle)],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_z_3d(
    angle: tf.Tensor, matrix: tf.Tensor
) -> tf.Tensor:
    rotation_x = tf.cast(
        [
            [tf.cos(angle), -tf.sin(angle), 0],
            [tf.sin(angle), tf.cos(angle), 0],
            [0, 0, 1],
        ],
        tf.float32,
    )
    return matrix @ rotation_x


@tf.function(experimental_follow_type_hints=True)
def create_matrix_rotation_2d(
    angle: tf.Tensor, matrix: tf.Tensor = None
) -> tf.Tensor:
    rotation = tf.cast(
        [[tf.cos(angle), -tf.sin(angle)], [tf.sin(angle), tf.cos(angle)]],
        tf.float32,
    )

    if matrix is None:
        return rotation
    return matrix @ rotation
    # return rotation


@tf.function(experimental_follow_type_hints=True)
def rotate_coords_3d(
    coords: tf.Tensor,
    angle_x: tf.Tensor,
    angle_y: tf.Tensor,
    angle_z: tf.Tensor,
) -> tf.Tensor:
    rot_matrix = tf.eye(tf.shape(coords)[0])

    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)

    return tf.reshape(
        tf.transpose(
            tf.transpose(tf.reshape(coords, (tf.shape(coords)[0], -1)))
            @ rot_matrix
        ),
        tf.shape(coords),
    )


@tf.function(experimental_follow_type_hints=True)
def rotate_coords_2d(coords: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    rot_matrix = create_matrix_rotation_2d(angle)
    print(tf.shape(coords))
    return tf.reshape(
        tf.transpose(
            tf.transpose(tf.reshape(coords, (tf.shape(coords)[0], -1)))
            @ rot_matrix
        ),
        tf.shape(coords),
    )


@tf.function(experimental_follow_type_hints=True)
def scale_coords(coords: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    return coords * tf.reshape(scale, (-1, 1))


# Gaussian filter related
@tf.function(
    autograph=False,
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    ],
)
def gaussian_kernel1d(sigma: tf.Tensor, radius: tf.Tensor) -> tf.Tensor:
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    phi = tf.exp(-0.5 / (sigma * sigma) * x ** 2)
    return phi / tf.reduce_sum(phi)


@tf.function(experimental_follow_type_hints=True)
def gf_pad(
    x: tf.Tensor,
    mode: tf.Tensor,
    cval: tf.Tensor,
    ws: tf.Tensor,
    ins: tf.Tensor,
):
    """Pad.

    for the VALID, width = (pin_w - k_w + 1) / stride(1) = in_w
    padding size = pin_w - in_w = k_w - 1
    Left geq Right

    mode 0: reflect
    mode 1: constant
    """
    pv = tf.cond(
        tf.equal(mode, 0),
        lambda: x,
        lambda: tf.zeros(ins) + cval,
    )
    lp = tf.concat([pv, pv[::-1]], axis=0)
    rp = tf.concat([pv[::-1], pv], axis=0)

    ll = (ws - 1 + 1) // 2
    rl = (ws - 1) - ll

    lpv = tf.tile(lp, [ll // 2 // ins + 1])[:ll][::-1]
    lrv = tf.tile(rp, [rl // 2 // ins + 1])[:rl]
    return tf.concat([lpv, x, lrv], axis=0)


@tf.function(experimental_follow_type_hints=True)
def gaussian_filter1d(
    xs: tf.Tensor,
    sigma: tf.Tensor,
    mode: tf.Tensor,
    cval: tf.Tensor = 0.0,
    truncate: tf.Tensor = 4.0,
):
    """Gaussian filter batched 1D.

    mode 0: reflect padding;
    mode 1: constant padding.
    """
    sigma = tf.cast(sigma, tf.float32)
    lw = tf.cast(truncate * sigma + 0.5, tf.int64)
    weights = gaussian_kernel1d(sigma, lw)[::-1]

    ws = tf.size(weights)
    ins = tf.shape(xs)[1]

    # padding
    xs = tf.map_fn(
        lambda x: gf_pad(
            x,
            mode,
            cval,
            ws,
            ins,
        ),
        xs,
    )

    xs = tf.reshape(xs, (-1, tf.shape(xs)[1], 1))
    kernel = tf.reshape(weights, (-1, 1, 1))

    return tf.squeeze(tf.nn.conv1d(xs, kernel, stride=1, padding="VALID"))


@tf.function(experimental_follow_type_hints=True, jit_compile=False)
def gaussian_filter(
    xxs: tf.Tensor,
    sigma: tf.Tensor,
    mode: tf.Tensor = 0,
    cval: tf.Tensor = 0.0,
    truncate: tf.Tensor = 4.0,
) -> tf.Tensor:
    """Gaussian filter trans from scipy gaussian filter.

    mode 0: reflect padding.
    mode 1: constant padding.

    NOTE: only for 3 dim Tensor.
    NOTE: jit false due to dynamic radius in kernel
    """
    xxs = tf.transpose(
        tf.map_fn(
            lambda xs: gaussian_filter1d(xs, sigma, mode, cval, truncate),
            tf.transpose(xxs, [2, 1, 0]),
        ),
        [2, 1, 0],
    )
    xxs = tf.transpose(
        tf.map_fn(
            lambda xs: gaussian_filter1d(xs, sigma, mode, cval, truncate),
            tf.transpose(xxs, [2, 0, 1]),
        ),
        [1, 2, 0],
    )
    return tf.transpose(
        tf.map_fn(
            lambda xs: gaussian_filter1d(xs, sigma, mode, cval, truncate),
            tf.transpose(xxs, [1, 0, 2]),
        ),
        [1, 0, 2],
    )


if __name__ == "__main__":
    # Others
    import scipy.ndimage.filters as sf

    patch_size = tf.constant([2, 3, 4])

    with tf.device("/CPU:0"):
        tf.print(get_range_val([0, 1.0]))
        coords = create_zero_centered_coordinate_mesh(patch_size)

        tf.print(elastic_deform_coordinates(coords, 50.0, 12.0).shape)
        # assert elastic_deform_coordinates(coords, 50., 12.).shape == [
        #     3,
        #     40,
        #     56,
        #     40,
        # ]

        xs = tf.random.uniform(patch_size, 0, 100)
        # s = xs.shape
        # tf.print(xs.shape)
        x = gaussian_filter(xs, 5, 0)
        tf.print("\n\n", x, "\n", x.shape, x.shape)
        x_ = sf.gaussian_filter(xs, 5, mode="reflect")
        tf.print("----\n", x_, "\n", x_.shape, x_.shape)

        # tf.print("----------")
        # tf.print(rotate_coords_3d(coords, 1.0, 1.0, 1.0).shape)

        tf.print(to_one_hot(tf.zeros([9, 40, 56, 40]), [1.0, 2, 3]).shape)
