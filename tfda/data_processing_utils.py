# @author Wenliang Zhong
# @email wenliang.zhong@uta.edu
# @create date 2021-12-14 18:42:00
# @modify date 2021-12-23 12:33:00
# @desc use for data loader and some processing

# Standard Library
import os
import pickle
from copy import deepcopy

# Tensorflow
import tensorflow as tf

# Types
from typing import Dict, Tuple

# Others
import tensorflow_addons as tfa
from tensorflow_graphics.math.interpolation import trilinear

# Local
from tfda.augmentations.utils import rotate_coords_2d, rotate_coords_3d
from tfda.defs import DTFT, TFbF, TFbT, TFDADefault3DParams, nan, pi
from tfda.utils import isnan


def get_batch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    rot_x = tf.reduce_max(tf.abs(rot_x))
    rot_y = tf.reduce_max(tf.abs(rot_y))
    rot_z = tf.reduce_max(tf.abs(rot_z))
    rot_x = tf.minimum(90 / 360 * 2.0 * pi, rot_x)
    rot_y = tf.minimum(90 / 360 * 2.0 * pi, rot_y)
    rot_z = tf.minimum(90 / 360 * 2.0 * pi, rot_z)
    coords = tf.cast(final_patch_size, dtype=tf.float32)
    final_shape = tf.identity(coords)
    if tf.shape(coords)[0] == 3:
        final_shape = tf.math.reduce_max(
            tf.stack(
                (
                    tf.abs(rotate_coords_3d(coords, rot_x, 0.0, 0.0)),
                    final_shape,
                )
            ),
            axis=0,
        )
        final_shape = tf.math.reduce_max(
            tf.stack(
                (
                    tf.abs(rotate_coords_3d(coords, 0.0, rot_y, 0.0)),
                    final_shape,
                )
            ),
            axis=0,
        )
        final_shape = tf.math.reduce_max(
            tf.stack(
                (
                    tf.abs(rotate_coords_3d(coords, 0.0, 0.0, rot_z)),
                    final_shape,
                )
            ),
            axis=0,
        )
    else:
        final_shape = tf.math.reduce_max(
            tf.stack((tf.abs(rotate_coords_2d(coords, rot_x)), final_shape)),
            axis=0,
        )
    final_shape /= tf.math.reduce_min(scale_range)
    return tf.cast(final_shape, tf.int32)


class DataAugmentor:
    def __init__(self, plans_files, jsn, plans=nan) -> nan:
        self.plans_file = plans_files
        self.jsn = jsn
        self.plans = plans
        self.threeD = nan
        self.do_dummy_2D_aug = nan
        self.use_mask_for_norm = nan
        self.basic_generator_patch_size = nan
        self.patch_size = nan
        self.batch_size = nan
        self.oversample_foregroung_percent = 0.33
        self.pad_all_sides = nan
        self.stage = nan
        self.data_aug_param = nan
        self.pseud_3d_slices = 1

    def initialize(self, training=TFbT, force_load_plans_file=TFbF):
        if force_load_plans_file or isnan(self.plans):
            self.load_plans_file()  #  here not sure whether pickle can be used in the tpu
        self.process_plans(self.plans)

        self.setup_DA_params()

    def load_plans_file(self):
        with open(self.plans_file, "rb") as f:
            self.plans = pickle.load(f)

    def transform_fn(self, dataset, input_context):
        dataset = dataset.batch(self.batch_size, drop_remainder=TFbT)
        dataset = dataset.map(
            self.formalize_data_3d,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return dataset

    def process_plans(self, plans):
        if isnan(self.stage):
            assert len(list(plans["plans_per_stage"].keys())) == 1, (
                "If self.stage is nan then there can be only one stage in the plans file. That seems to not be the "
                "case. Please specify which stage of the cascade must be trained"
            )
            self.stage = list(plans["plans_per_stage"].keys())[0]
        self.plans = plans

        stage_plans = self.plans["plans_per_stage"][self.stage]
        self.batch_size = stage_plans["batch_size"]
        self.patch_size = tf.convert_to_tensor(
            stage_plans["patch_size"], tf.int64
        )  # here, in the orginal code, the author convert it to np.array. It may need to change later.
        self.do_dummy_2D_aug = stage_plans["do_dumy_2D_data_aug"]
        self.pad_all_sides = nan
        self.use_mask_for_norm = plans["use_mask_for_norm"][
            0
        ]  # TODO here, the orignal vaule of this key is an Order Dict, which is not supported in TF Tensor. But since it has just one value, we just choose the first
        if len(self.patch_size) == 2:
            self.threeD = TFbF
        elif len(self.patch_size) == 3:
            self.threeD = TFbT
        else:
            raise RuntimeError(
                "invalid patch size in plans file: %s" % str(self.patch_size)
            )

    def setup_DA_params(self):
        if self.threeD:
            rotation_x = (
                -30.0 / 360 * 2.0 * pi,
                30.0 / 360 * 2.0 * pi,
            )
            rotation_y = (
                -30.0 / 360 * 2.0 * pi,
                30.0 / 360 * 2.0 * pi,
            )
            rotation_z = (
                -30.0 / 360 * 2.0 * pi,
                30.0 / 360 * 2.0 * pi,
            )
            if self.do_dummy_2D_aug:
                dummy_2D = TFbT
                # print("Using dummy2d data augmentation")
                elastic_deform_alpha = (0.0, 200.0)
                elastic_deform_sigma = (9.0, 13.0)
                rotation_x = (
                    -180.0 / 360 * 2.0 * pi,
                    180.0 / 360 * 2.0 * pi,
                )
                self.data_aug_param = TFDADefault3DParams(
                    rotation_x=rotation_x,
                    rotation_y=rotation_y,
                    rotation_z=rotation_z,
                    dummy_2D=dummy_2D,
                    elastic_deform_alpha=elastic_deform_alpha,
                    elastic_deform_sigma=elastic_deform_sigma,
                    scale_range=(0.7, 1.4),
                    do_elastic=TFbF,
                    selected_seg_channels=[0],
                    patch_size_for_spatial_transform=self.patch_size,
                    num_cached_per_thread=2,
                    mask_was_used_for_normalization=self.use_mask_for_norm,
                )
            else:
                self.data_aug_param = TFDADefault3DParams(
                    rotation_x=rotation_x,
                    rotation_y=rotation_y,
                    rotation_z=rotation_z,
                    scale_range=(0.7, 1.4),
                    do_elastic=TFbF,
                    selected_seg_channels=[0],
                    patch_size_for_spatial_transform=self.patch_size,
                    num_cached_per_thread=2,
                    mask_was_used_for_normalization=self.use_mask_for_norm,
                )
        else:
            self.do_dummy_2D_aug = TFbF
            if tf.maximum(self.patch_size) / tf.minimum(self.patch_size) > 1.5:
                rotation_x = (
                    -15.0 / 360 * 2.0 * pi,
                    15.0 / 360 * 2.0 * pi,
                )
            else:
                rotation_x = (
                    -180.0 / 360 * 2.0 * pi,
                    180.0 / 360 * 2.0 * pi,
                )
            elastic_deform_alpha = (0.0, 200.0)
            elastic_deform_sigma = (9.0, 13.0)
            rotation_y = (
                -0.0 / 360 * 2.0 * pi,
                0.0 / 360 * 2.0 * pi,
            )
            rotation_z = (
                -0.0 / 360 * 2.0 * pi,
                0.0 / 360 * 2.0 * pi,
            )
            dummy_2D = TFbF
            mirror_axes = (
                0,
                1,
            )
            self.data_aug_param = TFDADefault3DParams(
                rotation_x=rotation_x,
                rotation_y=rotation_y,
                rotation_z=rotation_z,
                elastic_deform_alpha=elastic_deform_alpha,
                elastic_deform_sigma=elastic_deform_sigma,
                dummy_2D=dummy_2D,
                mirror_axes=mirror_axes,
                mask_was_used_for_normalization=self.use_mask_for_norm,
                scale_range=(0.7, 1.4),
                do_elastic=TFbF,
                selected_seg_channels=[0],
                patch_size_for_spatial_transform=self.patch_size,
                num_cached_per_thread=2,
            )

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_batch_size(
                self.patch_size[1:],
                self.data_aug_param["rotation_x"],
                self.data_aug_param["rotation_y"],
                self.data_aug_param["rotation_z"],
                (0.85, 1.25),
            )
            self.basic_generator_patch_size = tf.constant(
                [self.patch_size[0]] + list(self.basic_generator_patch_size)
            )

        else:
            self.basic_generator_patch_size = get_batch_size(
                self.patch_size,
                self.data_aug_param["rotation_x"],
                self.data_aug_param["rotation_y"],
                self.data_aug_param["rotation_z"],
                (0.85, 1.25),
            )
        self.basic_generator_patch_size = tf.cast(
            self.basic_generator_patch_size, tf.int64
        )

    @tf.function(experimental_follow_type_hints=TFbT)
    def formalize_data_3d(self, data: DTFT) -> Tuple[tf.Tensor, tf.Tensor]:

        # main body begin here
        (
            case_identifier,
            image_raw,
            label_raw,
            class_locations_bytes,
            class_locations_shape,
        ) = (
            data["case_identifier"],
            data["image/encoded"],
            data["image/class/label"],
            data["class_locations_bytes"],
            data["class_locations_shape"],
        )
        # TODO please note that in this "generate_train_batch" function, we don't deal the 3D UNet Cascade. We will modify it after we tune the pipeline of casecade
        original_image_size = tf.cast(data["image/shape"], dtype=tf.int64)
        original_label_size = tf.cast(data["label/shape"], dtype=tf.int64)

        results = tf.map_fn(
            lambda i: process_batch(
                i,
                image_raw,
                original_image_size,
                original_label_size,
                label_raw,
                class_locations_bytes,
                class_locations_shape,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                self.oversample_foregroung_percent,
            ),
            elems=tf.range(self.batch_size, dtype=tf.float32),
        )
        images = results[:, 0]
        segs = results[:, 1]
        # assert data is nan, f'{images}'
        # data["images"] = images
        # data["labels"] = segs
        # return data
        images, segs = tf.transpose(images, (0, 2, 3, 4, 1)), tf.transpose(
            segs, (0, 2, 3, 4, 1)
        )
        return images, segs


@tf.function
def get_do_oversample(batch_idx, batch_size, oversample_foregroung_percent):
    # return not batch_idx < round(self.batch_size * (1 - self.oversample_foregroung_percent))
    return tf.greater_equal(
        tf.cast(batch_idx, tf.float32),
        tf.round(
            tf.cast(batch_size, tf.float32)
            * (tf.cast(1 - oversample_foregroung_percent, tf.float32))
        ),
    )


@tf.function
def update_need_to_pad(
    need_to_pad, d, basic_generator_patch_size, case_all_data
):
    need_to_pad_d = (
        basic_generator_patch_size[d]
        - tf.shape(case_all_data, out_type=tf.int64)[d + 1]
    )
    return tf.cond(
        tf.less(
            need_to_pad[d] + tf.shape(case_all_data, out_type=tf.int64)[d + 1],
            basic_generator_patch_size[d],
        ),
        lambda: need_to_pad_d,
        lambda: need_to_pad[d],
    )


@tf.function
def not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z):
    bbox_x_lb = tf.random.uniform(
        [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
    )
    bbox_y_lb = tf.random.uniform(
        [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
    )
    bbox_z_lb = tf.random.uniform(
        [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
    )
    return bbox_x_lb, bbox_y_lb, bbox_z_lb


@tf.function
def do_force_fg_selected(
    lb_x, lb_y, lb_z, selected_voxel, basic_generator_patch_size
):
    bbox_x_lb = tf.maximum(
        lb_x,
        selected_voxel[0] - basic_generator_patch_size[0] // 2,
    )
    bbox_y_lb = tf.maximum(
        lb_y,
        selected_voxel[1] - basic_generator_patch_size[1] // 2,
    )
    bbox_z_lb = tf.maximum(
        lb_z,
        selected_voxel[2] - basic_generator_patch_size[2] // 2,
    )
    return bbox_x_lb, bbox_y_lb, bbox_z_lb


@tf.function
def do_force_fg(
    i,
    class_locations_types,
    class_locations_shape,
    class_locations_bytes,
    lb_x,
    ub_x,
    lb_y,
    ub_y,
    lb_z,
    ub_z,
    basic_generator_patch_size,
):
    c = random_choice(class_locations_types, 0)[0]
    class_locations_decode = tf.io.decode_raw(
        class_locations_bytes[i][c], tf.int64
    )
    class_locations = tf.reshape(
        class_locations_decode, [class_locations_shape[i][c], -1]
    )
    selected_voxel = random_choice(class_locations, 0)[0]

    return tf.cond(
        isnan(tf.cast(selected_voxel, tf.float32)),
        lambda: not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z),
        lambda: do_force_fg_selected(
            lb_x, lb_y, lb_z, selected_voxel, basic_generator_patch_size
        ),
    )
    # return do_force_fg_selected(lb_x, lb_y, lb_z, selected_voxel, basic_generator_patch_size)
    # return not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z)


def transform_fn_wrapper(
    basic_generator_patch_size,
    patch_size,
    batch_size,
    oversample_foregroung_percent,
):
    def transform_fn_fn(dataset, input_context):
        return transform_fn_2(
            dataset,
            basic_generator_patch_size,
            patch_size,
            batch_size,
            oversample_foregroung_percent,
        )

    return transform_fn_fn


def transform_fn_2(
    dataset,
    basic_generator_patch_size,
    patch_size,
    batch_size,
    oversample_foregroung_percent,
):
    dataset = dataset.batch(batch_size, drop_remainder=TFbT)
    dataset = dataset.map(
        formalize_data_3d_wrapper(
            basic_generator_patch_size,
            patch_size,
            batch_size,
            oversample_foregroung_percent,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def formalize_data_3d_wrapper(
    basic_generator_patch_size,
    patch_size,
    batch_size,
    oversample_foregroung_percent,
):
    @tf.function
    def formalize_data_3d_fn(dataset):
        return formalize_data_3d_2(
            dataset,
            basic_generator_patch_size,
            patch_size,
            batch_size,
            oversample_foregroung_percent,
        )

    return formalize_data_3d_fn


@tf.function(experimental_follow_type_hints=TFbT)
def formalize_data_3d_2(
    data: DTFT,
    basic_generator_patch_size,
    patch_size,
    batch_size,
    oversample_foregroung_percent,
) -> Tuple[tf.Tensor, tf.Tensor]:

    # main body begin here
    (
        case_identifier,
        image_raw,
        label_raw,
        class_locations_bytes,
        class_locations_shape,
    ) = (
        data["case_identifier"],
        data["image/encoded"],
        data["image/class/label"],
        data["class_locations_bytes"],
        data["class_locations_shape"],
    )
    # TODO please note that in this "generate_train_batch" function, we don't deal the 3D UNet Cascade. We will modify it after we tune the pipeline of casecade
    original_image_size = tf.cast(data["image/shape"], dtype=tf.int64)
    original_label_size = tf.cast(data["label/shape"], dtype=tf.int64)

    results = tf.map_fn(
        lambda i: process_batch(
            i,
            image_raw,
            original_image_size,
            original_label_size,
            label_raw,
            class_locations_bytes,
            class_locations_shape,
            basic_generator_patch_size,
            patch_size,
            batch_size,
            oversample_foregroung_percent,
        ),
        elems=tf.range(batch_size, dtype=tf.float32),
    )
    images = results[:, 0]
    segs = results[:, 1]
    # assert data is nan, f'{images}'
    # data["images"] = images
    # data["labels"] = segs
    # return data
    images, segs = tf.transpose(images, (0, 2, 3, 4, 1)), tf.transpose(
        segs, (0, 2, 3, 4, 1)
    )
    return images, segs


@tf.function
def process_batch(
    ii,
    image_raw,
    original_image_size,
    original_label_size,
    label_raw,
    class_locations_bytes,
    class_locations_shape,
    basic_generator_patch_size,
    patch_size,
    batch_size,
    oversample_foregroung_percent,
):
    i = tf.cast(ii, tf.int64)
    zero = tf.constant(0, dtype=tf.int64)
    image = tf.io.decode_raw(image_raw[i], tf.as_dtype(tf.float32))
    label = tf.io.decode_raw(label_raw[i], tf.as_dtype(tf.float32))
    original_image, original_label = tf.reshape(
        image, original_image_size[i]
    ), tf.reshape(label, original_label_size[i])
    image = tf.cast(original_image, dtype=tf.float32)
    label = tf.cast(original_label, dtype=tf.float32)

    # assert class_locations_bytes is nan, f'{}'
    class_locations_types = tf.range(tf.shape(class_locations_bytes[i])[0])

    # just for test here
    # img = volume_resize(image[0], basic_generator_patch_size, 'bicubic')
    # seg = volume_resize(label[0], basic_generator_patch_size, 'bicubic')
    # result = tf.stack([img, seg])
    # tf.print(tf.shape(result))
    # return result

    case_all_data = tf.concat([image, label], axis=0)
    # tf.print(tf.shape(case_all_data))
    force_fg = tf.less(
        tf.cast(i, tf.float32),
        tf.round(
            tf.cast(batch_size, tf.float32)
            * (tf.cast(1 - oversample_foregroung_percent, tf.float32))
        ),
    )
    basic_generator_patch_size = tf.cast(
        basic_generator_patch_size, dtype=tf.int64
    )
    patch_size = tf.cast(patch_size, dtype=tf.int64)
    need_to_pad = basic_generator_patch_size - patch_size
    need_to_pad = tf.map_fn(
        lambda d: update_need_to_pad(
            need_to_pad, d, basic_generator_patch_size, case_all_data
        ),
        elems=tf.range(3, dtype=tf.int64),
    )
    need_to_pad = tf.cast(need_to_pad, tf.int64)
    shape = tf.shape(case_all_data, out_type=tf.int64)[1:]
    lb_x = -need_to_pad[0] // 2
    ub_x = (
        shape[0]
        + need_to_pad[0] // 2
        + need_to_pad[0] % 2
        - basic_generator_patch_size[0]
    )
    lb_y = -need_to_pad[1] // 2
    ub_y = (
        shape[1]
        + need_to_pad[1] // 2
        + need_to_pad[1] % 2
        - basic_generator_patch_size[1]
    )
    lb_z = -need_to_pad[2] // 2
    ub_z = (
        shape[2]
        + need_to_pad[2] // 2
        + need_to_pad[2] % 2
        - basic_generator_patch_size[2]
    )

    bbox_x_lb, bbox_y_lb, bbox_z_lb = tf.cond(
        force_fg,
        lambda: not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z),
        lambda: do_force_fg(
            i,
            class_locations_types,
            class_locations_shape,
            class_locations_bytes,
            lb_x,
            ub_x,
            lb_y,
            ub_y,
            lb_z,
            ub_z,
            basic_generator_patch_size,
        ),
    )

    bbox_x_ub = bbox_x_lb + basic_generator_patch_size[0]
    bbox_y_ub = bbox_y_lb + basic_generator_patch_size[1]
    bbox_z_ub = bbox_z_lb + basic_generator_patch_size[2]

    valid_bbox_x_lb = tf.maximum(zero, bbox_x_lb)
    valid_bbox_x_ub = tf.minimum(shape[0], bbox_x_ub)
    valid_bbox_y_lb = tf.maximum(zero, bbox_y_lb)
    valid_bbox_y_ub = tf.minimum(shape[1], bbox_y_ub)
    valid_bbox_z_lb = tf.maximum(zero, bbox_z_lb)
    valid_bbox_z_ub = tf.minimum(shape[2], bbox_z_ub)

    case_all_data = tf.identity(
        case_all_data[
            :,
            valid_bbox_x_lb:valid_bbox_x_ub,
            valid_bbox_y_lb:valid_bbox_y_ub,
            valid_bbox_z_lb:valid_bbox_z_ub,
        ]
    )

    img = tf.pad(
        case_all_data[:-1],
        (
            [0, 0],
            [
                -tf.minimum(zero, bbox_x_lb),
                tf.maximum(bbox_x_ub - shape[0], zero),
            ],
            [
                -tf.minimum(zero, bbox_y_lb),
                tf.maximum(bbox_y_ub - shape[1], zero),
            ],
            [
                -tf.minimum(zero, bbox_z_lb),
                tf.maximum(bbox_z_ub - shape[2], zero),
            ],
        ),
        mode="CONSTANT",
    )
    seg = tf.pad(
        case_all_data[-1:],
        (
            [0, 0],
            [
                -tf.minimum(zero, bbox_x_lb),
                tf.maximum(bbox_x_ub - shape[0], zero),
            ],
            [
                -tf.minimum(zero, bbox_y_lb),
                tf.maximum(bbox_y_ub - shape[1], zero),
            ],
            [
                -tf.minimum(zero, bbox_z_lb),
                tf.maximum(bbox_z_ub - shape[2], zero),
            ],
        ),
        mode="CONSTANT",
        constant_values=-1,
    )
    result = tf.stack([img, seg])
    # tf.print(tf.shape(result))
    return result


@tf.function
def augment_spatial(
    data,
    seg,
    patch_size,
    patch_center_dist_from_border=30,
    do_elastic_deform=TFbT,
    alpha=(0.0, 1000.0),
    sigma=(10.0, 13.0),
    do_rotation=TFbT,
    angle_x=(0, 2 * tf.constant(pi)),
    angle_y=(0, 2 * tf.constant(pi)),
    angle_z=(0, 2 * tf.constant(pi)),
    do_scale=TFbT,
    scale=(0.75, 1.25),
    border_mode_data="nearest",
    border_cval_data=0,
    order_data=3,
    border_mode_seg="constant",
    border_cval_seg=0,
    order_seg=3,
    random_crop=TFbT,
    p_el_per_sample=1.0,
    p_scale_per_sample=1.0,
    p_rot_per_sample=1,
    independent_scale_for_each_axis=TFbF,
    p_rot_per_axis: float = 1.0,
    p_independent_scale_per_axis: int = 1,
):
    def augment_per_sample(
        sample_id, patch_size, data, seg, data_result, seg_result
    ):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = TFbT
        if modified_coords:

            d = tf.constant(0)
            loop_cond = lambda d, coords: tf.less(d, dim)

            def body_fn(d, coords):
                if random_crop:
                    ctr = tf.random.uniform(
                        [],
                        patch_center_dist_from_border[d],
                        tf.cast(tf.shape(data)[d + 2], dtype=tf.float32)
                        - patch_center_dist_from_border[d],
                    )
                else:
                    ctr = (
                        tf.cast(tf.shape(data)[d + 2], dtype=tf.float32) / 2.0
                        - 0.5
                    )
                coords_d = coords[d] + ctr
                coords = update_tf_channel(coords, d, coords_d)
                d = d + 1
                coords.set_shape([3, nan, nan, nan])
                return d, coords

            _, coords = tf.while_loop(
                loop_cond,
                body_fn,
                [d, coords],
                shape_invariants=[tf.TensorShape(nan), coords.get_shape()],
            )
            data_sample = tf.zeros(tf.shape(data_result)[1:])
            channel_id = tf.constant(0)
            cond_to_loop_data = lambda channel_id, data_sample: tf.less(
                channel_id, tf.shape(data)[1]
            )

            def body_fn_data(channel_id, data_sample):
                data_channel = interpolate_img(
                    data[sample_id, channel_id],
                    coords,
                    order_data,
                    border_mode_data,
                    border_cval_data,
                )
                data_sample = update_tf_channel(
                    data_sample, channel_id, data_channel
                )
                channel_id = channel_id + 1
                return channel_id, data_sample

            _, data_sample = tf.while_loop(
                cond_to_loop_data, body_fn_data, [channel_id, data_sample]
            )
            data_result = update_tf_channel(
                data_result, sample_id, data_sample
            )
            if seg is not nan:
                seg_sample = tf.zeros(tf.shape(seg_result)[1:])
                channel_id = tf.constant(0)
                cond_to_loop_seg = lambda channel_id, seg_sample: tf.less(
                    channel_id, tf.shape(seg)[1]
                )

                def body_fn_seg(channel_id, seg_sample):
                    seg_channel = interpolate_img(
                        seg[sample_id, channel_id],
                        coords,
                        order_seg,
                        border_mode_seg,
                        border_cval_seg,
                        is_seg=TFbT,
                    )
                    seg_sample = update_tf_channel(
                        seg_sample, channel_id, seg_channel
                    )
                    channel_id = channel_id + 1
                    return channel_id, seg_sample

                _, seg_sample = tf.while_loop(
                    cond_to_loop_seg, body_fn_seg, [channel_id, seg_sample]
                )
                seg_result = update_tf_channel(
                    seg_result, sample_id, seg_sample
                )
        else:
            if isnan(seg):
                s = nan
            else:
                s = seg[sample_id : sample_id + 1]
            if random_crop:
                # margin = [patch_center_dist_from_border[d] - tf.cast(patch_size[d], dtype=tf.float32) // 2 for d in tf.range(dim)]
                margin = tf.map_fn(
                    lambda d: tf.cast(
                        patch_center_dist_from_border[d], dtype=tf.int64
                    )
                    - patch_size[d] // 2,
                    elems=tf.range(dim),
                )
                d, s = random_crop_fn(
                    data[sample_id : sample_id + 1], s, patch_size, margin
                )
            else:
                d, s = center_crop_fn(
                    data[sample_id : sample_id + 1], patch_size, s
                )
            data_result = update_tf_channel(data_result, sample_id, d[0])
            if seg is not nan:
                seg_result = update_tf_channel(seg_result, sample_id, s[0])
        sample_id = sample_id + 1
        return sample_id, patch_size, data, seg, data_result, seg_result

    if not isinstance(patch_size, tf.Tensor):
        patch_size = tf.convert_to_tensor(patch_size)
    if not isinstance(patch_center_dist_from_border, tf.Tensor):
        patch_center_dist_from_border = tf.convert_to_tensor(
            patch_center_dist_from_border, dtype=tf.float32
        )
    cond_to_loop = lambda sample_id, patch_size, data, seg, data_result, seg_result: tf.less(
        sample_id, sample_num
    )
    dim = tf.shape(patch_size)[0]
    seg_result = nan
    if seg is not nan:
        seg_result = tf.cond(
            tf.equal(dim, tf.constant(2)),
            lambda: tf.zeros(
                tf.concat([tf.shape(seg)[:2], patch_size[:2]], axis=0)
            ),
            lambda: tf.zeros(
                tf.concat([tf.shape(seg)[:2], patch_size[:3]], axis=0)
            ),
        )

    data_result = tf.cond(
        tf.equal(dim, tf.constant(2)),
        lambda: tf.zeros(
            tf.concat([tf.shape(data)[:2], patch_size[:2]], axis=0)
        ),
        lambda: tf.zeros(
            tf.concat([tf.shape(data)[:2], patch_size[:3]], axis=0)
        ),
    )
    sample_num = tf.shape(data)[0]
    sample_id = tf.constant(0)
    _, _, _, _, data_result, seg_result = tf.while_loop(
        cond_to_loop,
        augment_per_sample,
        [sample_id, patch_size, data, seg, data_result, seg_result],
    )
    return data_result, seg_result


@tf.function
def interpolate_img(
    img, coords, order=3, mode="nearest", cval=0.0, is_seg=TFbF
):
    unique_labels, _ = tf.unique(tf.reshape(img, (1, -1))[0])
    if is_seg and order != 0:
        # assert img is nan, f'{img}'
        result = tf.zeros(tf.shape(coords)[1:], dtype=tf.float32)
        cond_to_loop = lambda img, i, coords, result, order: tf.less(
            i, tf.shape(unique_labels)[0]
        )

        def body_fn(img, i, coords, result, order):
            img, _, coords, result, order = map_coordinates_seg(
                img, unique_labels[i], coords, result, 3
            )  # here I force the order = 3
            i = i + 1
            return img, i, coords, result, order

        i = tf.constant(0)
        _, _, _, result, _ = tf.while_loop(
            cond_to_loop, body_fn, [img, i, coords, result, order]
        )
        return result
    else:
        return map_coordinates_img(img, coords, 3)


@tf.function
def map_coordinates_seg(seg, cl, coords, result, order):
    cl_seg = tf.cast(tf.equal(seg, cl), dtype=tf.float32)
    # order = tf.cast(order, tf.int64)
    new_seg = tf.cond(
        tf.equal(tf.rank(seg), tf.constant(3)),
        lambda: map_linear_coordinates_3d(cl_seg, coords),
        lambda: map_coordinates_2d(cl_seg, coords, order),
    )
    indices = tf.where(tf.greater_equal(new_seg, tf.constant(0.5)))
    result = tf.tensor_scatter_nd_update(
        result, indices, tf.ones(tf.shape(indices)[0]) * cl
    )
    return seg, cl, coords, result, order


@tf.function
def map_coordinates_img(img, coords, order=3):
    # return tf.cond(tf.equal(tf.rank(img), tf.constant(3)), lambda: map_coordinates_3d(img, coords, order), lambda: map_coordinates_2d(img, coords, order))
    return tf.cond(
        tf.equal(tf.rank(img), tf.constant(3)),
        lambda: map_linear_coordinates_3d(img, coords),
        lambda: map_coordinates_2d(img, coords, order),
    )


@tf.function
def map_random_coordinates_3d(img, coords, order=3):
    new_coords = tf.concat(
        [
            tf.reshape(coords[0], (-1, 1)),
            tf.reshape(coords[1], (-1, 1)),
            tf.reshape(coords[2], (-1, 1)),
        ],
        axis=1,
    )
    new_coords = new_coords[
        tf.newaxis,
    ]
    new_coords = tf.cast(new_coords, tf.float32)
    x, y, z = tf.meshgrid(
        tf.range(tf.shape(img)[0]),
        tf.range(tf.shape(img)[1]),
        tf.range(tf.shape(img)[2]),
        indexing="ij",
    )
    x, y, z = (
        tf.reshape(x, (-1, 1)),
        tf.reshape(y, (-1, 1)),
        tf.reshape(z, (-1, 1)),
    )
    original_coords = tf.concat([x, y, z], axis=1)
    original_coords = original_coords[
        tf.newaxis,
    ]
    original_coords = tf.cast(original_coords, tf.float32)
    random_coordinates_len = tf.cast(tf.round(tf.shape(new_coords)[1] / 7), tf.int32)
    original_coordinates_len = tf.shape(original_coords)[1]
    random_indexes = random_sample(tf.range(original_coordinates_len), 0, random_coordinates_len)
    random_indexes = tf.sort(random_indexes)
    random_coords = tf.gather(original_coords, random_indexes, axis=1)
    tmp_img = tf.reshape(img, (1, -1, 1))
    tmp_img = tf.cast(tmp_img, tf.float32)
    random_img = tf.gather(tmp_img, random_indexes, axis=1)
    result = tfa.image.interpolate_spline(
        random_coords, random_img, new_coords, order=order
    )
    result = tf.reshape(result, tf.shape(coords)[1:])
    return result

@tf.function
def map_linear_coordinates_3d(img, coords):
    # tf.print(tf.shape(img))
    # tf.print(tf.shape(coords))
    new_coords = tf.concat(
        [
            tf.reshape(coords[0], (-1, 1)),
            tf.reshape(coords[1], (-1, 1)),
            tf.reshape(coords[2], (-1, 1)),
        ],
        axis=1,
    )
    new_coords = new_coords[
        tf.newaxis,
    ]
    new_coords = tf.cast(new_coords, tf.float32)
    tmp_img = img[:, :, :, tf.newaxis]
    tmp_img = tf.cast(tmp_img[tf.newaxis], tf.float32)
    result = trilinear.interpolate(
        tmp_img, new_coords
    )
    result = tf.reshape(result, tf.shape(coords)[1:])
    return result

@tf.function
def map_coordinates_3d(img, coords, order=3):
    new_coords = tf.concat(
        [
            tf.reshape(coords[0], (-1, 1)),
            tf.reshape(coords[1], (-1, 1)),
            tf.reshape(coords[2], (-1, 1)),
        ],
        axis=1,
    )
    new_coords = new_coords[
        tf.newaxis,
    ]
    new_coords = tf.cast(new_coords, tf.float32)
    x, y, z = tf.meshgrid(
        tf.range(tf.shape(img)[0]),
        tf.range(tf.shape(img)[1]),
        tf.range(tf.shape(img)[2]),
        indexing="ij",
    )
    x, y, z = (
        tf.reshape(x, (-1, 1)),
        tf.reshape(y, (-1, 1)),
        tf.reshape(z, (-1, 1)),
    )
    original_coords = tf.concat([x, y, z], axis=1)
    original_coords = original_coords[
        tf.newaxis,
    ]
    original_coords = tf.cast(original_coords, tf.float32)
    tmp_img = tf.reshape(img, (1, -1, 1))
    tmp_img = tf.cast(tmp_img, tf.float32)
    result = tfa.image.interpolate_spline(
        original_coords, tmp_img, new_coords, order=order
    )
    result = tf.reshape(result, tf.shape(coords)[1:])
    return result


@tf.function
def cond_to_loop_i(
    i, chunk_index, chunk_shape, total_result, chunk_size, coords, img
):
    return tf.less(i, chunk_size)


@tf.function
def body_fn_i(
    i, chunk_index, chunk_shape, total_result, chunk_size, coords, img
):
    chunk_coords = tf.slice(coords, chunk_index, chunk_shape)
    chunk_coords_0, chunk_coords_1, chunk_coords_2 = (
        tf.reshape(chunk_coords[0], (-1, 1)),
        tf.reshape(chunk_coords[1], (-1, 1)),
        tf.reshape(chunk_coords[2], (-1, 1)),
    )
    chunk_coords = tf.concat(
        [chunk_coords_0, chunk_coords_1, chunk_coords_2], axis=1
    )
    chunk_min = tf.math.reduce_min(chunk_coords, axis=0)
    chunk_max = tf.math.reduce_max(chunk_coords, axis=0)
    chunk_min, chunk_max = tf.cast(
        tf.maximum(tf.constant(0.0), tf.floor(chunk_min)), tf.int64
    ), tf.cast(
        tf.minimum(
            tf.cast(tf.shape(img), dtype=tf.float32),
            tf.round(chunk_max),
        ),
        tf.int64,
    )
    slice_size = chunk_max - chunk_min
    slice_img = tf.slice(img, chunk_min, slice_size)

    @tf.function
    def do_update(total_result, chunk_coords):
        reshape_img = tf.reshape(slice_img, (1, -1, 1))
        slice_x, slice_y, slice_z = tf.meshgrid(
            tf.range(chunk_min[0], chunk_max[0]),
            tf.range(chunk_min[1], chunk_max[1]),
            tf.range(chunk_min[2], chunk_max[2]),
            indexing="ij",
        )
        slice_x = tf.reshape(slice_x, (-1, 1))
        slice_y = tf.reshape(slice_y, (-1, 1))
        slice_z = tf.reshape(slice_z, (-1, 1))
        slice_coords = tf.concat([slice_x, slice_y, slice_z], axis=1)
        slice_coords = tf.cast(
            slice_coords[
                tf.newaxis,
            ],
            tf.float32,
        )
        chunk_coords = tf.cast(
            chunk_coords[
                tf.newaxis,
            ],
            tf.float32,
        )
        order = tf.constant(3, dtype=tf.int64)
        result = tfa.image.interpolate_spline(
            slice_coords, reshape_img, chunk_coords, order=3
        )
        result = result[:, :, 0]
        x, y, z = tf.meshgrid(
            tf.range(chunk_index[1], chunk_index[1] + chunk_shape[1]),
            tf.range(chunk_index[2], chunk_index[2] + chunk_shape[2]),
            tf.range(chunk_index[3], chunk_index[3] + chunk_shape[3]),
            indexing="ij",
        )
        x, y, z = (
            tf.reshape(x, (-1, 1)),
            tf.reshape(y, (-1, 1)),
            tf.reshape(z, (-1, 1)),
        )
        xyz = tf.concat([x, y, z], axis=1)
        map_coords = xyz[
            tf.newaxis,
        ]
        total_result = tf.tensor_scatter_nd_add(
            total_result, map_coords, result
        )
        return total_result

    total_result = tf.cond(
        tf.logical_and(
            tf.equal(tf.math.reduce_min(slice_img), tf.constant(0.0)),
            tf.equal(tf.math.reduce_max(slice_img), tf.constant(0.0)),
        ),
        lambda: total_result,
        lambda: do_update(total_result, chunk_coords),
    )

    chunk_index = tf.tensor_scatter_nd_add(
        chunk_index, [[1]], [chunk_shape[1]]
    )
    # chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[0]], [0])
    chunk_shape = tf.cond(
        tf.equal(i, chunk_size - 2),
        lambda: tf.tensor_scatter_nd_update(
            chunk_shape,
            [[1]],
            [tf.shape(coords, out_type=tf.int64)[1] - chunk_index[1]],
        ),
        lambda: chunk_shape,
    )
    i = i + 1
    return i, chunk_index, chunk_shape, total_result, chunk_size, coords, img


@tf.function
def cond_to_loop_j(
    j, chunk_index, chunk_shape, total_result, chunk_size, coords, img
):
    return tf.less(j, chunk_size)


@tf.function
def body_fn_j(
    j, chunk_index, chunk_shape, total_result, chunk_size, coords, img
):
    i = tf.constant(0, dtype=tf.int64)
    (
        _,
        chunk_index,
        chunk_shape,
        total_result,
        chunk_size,
        coords,
        img,
    ) = tf.while_loop(
        cond_to_loop_i,
        body_fn_i,
        [i, chunk_index, chunk_shape, total_result, chunk_size, coords, img],
    )
    chunk_index = tf.tensor_scatter_nd_add(
        chunk_index, [[2]], [chunk_shape[2]]
    )
    chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[1]], [0])
    chunk_shape = tf.tensor_scatter_nd_update(
        chunk_shape,
        [[1]],
        [tf.shape(coords, out_type=tf.int64)[1] // chunk_size],
    )
    chunk_shape = tf.cond(
        tf.equal(j, chunk_size - 2),
        lambda: tf.tensor_scatter_nd_update(
            chunk_shape,
            [[2]],
            [tf.shape(coords, out_type=tf.int64)[2] - chunk_index[2]],
        ),
        lambda: chunk_shape,
    )
    j = j + 1
    return j, chunk_index, chunk_shape, total_result, chunk_size, coords, img


@tf.function
def cond_to_loop_k(
    k, chunk_index, chunk_shape, total_result, chunk_size, coords, img
):
    return tf.less(k, chunk_size)


@tf.function
def body_fn_k(
    k, chunk_index, chunk_shape, total_result, chunk_size, coords, img
):
    j = tf.constant(0, dtype=tf.int64)
    (
        _,
        chunk_index,
        chunk_shape,
        total_result,
        chunk_size,
        coords,
        img,
    ) = tf.while_loop(
        cond_to_loop_j,
        body_fn_j,
        [j, chunk_index, chunk_shape, total_result, chunk_size, coords, img],
    )
    chunk_index = tf.tensor_scatter_nd_add(
        chunk_index, [[3]], [chunk_shape[3]]
    )
    chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[2]], [0])
    chunk_shape = tf.tensor_scatter_nd_update(
        chunk_shape,
        [[2]],
        [tf.cast(tf.shape(coords)[2], dtype=tf.int64) // chunk_size],
    )
    chunk_shape = tf.cond(
        tf.equal(k, chunk_size - 2),
        lambda: tf.tensor_scatter_nd_update(
            chunk_shape,
            [[3]],
            [tf.shape(coords, out_type=tf.int64)[3] - chunk_index[3]],
        ),
        lambda: chunk_shape,
    )
    k = k + 1
    return k, chunk_index, chunk_shape, total_result, chunk_size, coords, img


@tf.function
def map_chunk_coordinates_3d_2(img, coords, order=3, chunk_size=4):
    chunk_size = tf.constant(chunk_size, dtype=tf.int64)
    chunk_shape = tf.cast(tf.shape(coords)[1:], dtype=tf.int64) // chunk_size
    chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
    chunk_shape = tf.cast(chunk_shape, tf.int64)
    chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
    total_result = tf.zeros(tf.shape(coords)[1:])
    k = tf.constant(0, dtype=tf.int64)
    (
        _,
        chunk_index,
        chunk_shape,
        total_result,
        chunk_size,
        coords,
        img,
    ) = tf.while_loop(
        cond_to_loop_k,
        body_fn_k,
        [k, chunk_index, chunk_shape, total_result, chunk_size, coords, img],
    )
    return total_result


@tf.function
def map_chunk_coordinates_3d(img, coords, order=3, chunk_size=4):
    img = tf.cond(
        tf.equal(tf.rank(img), tf.constant(3)),
        lambda: img,
        lambda: tf.zeros((8, 8, 8)),
    )
    chunk_shape = tf.shape(coords)[1:] // chunk_size
    chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
    chunk_shape = tf.cast(chunk_shape, tf.int64)
    chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
    total_result = tf.zeros(tf.shape(coords)[1:])
    cond_to_loop_k = lambda k, chunk_index, chunk_shape, total_result: tf.less(
        k, tf.constant(chunk_size)
    )

    def body_fn_k(k, chunk_index, chunk_shape, total_result):
        cond_to_loop_j = (
            lambda j, chunk_index, chunk_shape, total_result: tf.less(
                j, tf.constant(chunk_size)
            )
        )

        def body_fn_j(j, chunk_index, chunk_shape, total_result):
            cond_to_loop_i = (
                lambda i, chunk_index, chunk_shape, total_result: tf.less(
                    i, chunk_size
                )
            )

            def body_fn_i(i, chunk_index, chunk_shape, total_result):
                chunk_coords = tf.slice(coords, chunk_index, chunk_shape)
                chunk_coords_0, chunk_coords_1, chunk_coords_2 = (
                    tf.reshape(chunk_coords[0], (-1, 1)),
                    tf.reshape(chunk_coords[1], (-1, 1)),
                    tf.reshape(chunk_coords[2], (-1, 1)),
                )
                chunk_coords = tf.concat(
                    [chunk_coords_0, chunk_coords_1, chunk_coords_2], axis=1
                )
                chunk_min = tf.math.reduce_min(chunk_coords, axis=0)
                chunk_max = tf.math.reduce_max(chunk_coords, axis=0)
                chunk_min, chunk_max = tf.cast(
                    tf.maximum(tf.constant(0.0), tf.floor(chunk_min)), tf.int64
                ), tf.cast(
                    tf.minimum(
                        tf.cast(tf.shape(img), dtype=tf.float32),
                        tf.round(chunk_max),
                    ),
                    tf.int64,
                )
                slice_size = chunk_max - chunk_min
                slice_img = tf.slice(img, chunk_min, slice_size)
                reshape_img = tf.reshape(slice_img, (1, -1, 1))
                slice_x, slice_y, slice_z = tf.meshgrid(
                    tf.range(chunk_min[0], chunk_max[0]),
                    tf.range(chunk_min[1], chunk_max[1]),
                    tf.range(chunk_min[2], chunk_max[2]),
                    indexing="ij",
                )
                slice_x = tf.reshape(slice_x, (-1, 1))
                slice_y = tf.reshape(slice_y, (-1, 1))
                slice_z = tf.reshape(slice_z, (-1, 1))
                slice_coords = tf.concat([slice_x, slice_y, slice_z], axis=1)
                slice_coords = tf.cast(
                    slice_coords[
                        tf.newaxis,
                    ],
                    tf.float32,
                )
                chunk_coords = tf.cast(
                    chunk_coords[
                        tf.newaxis,
                    ],
                    tf.float32,
                )
                order = tf.constant(3, dtype=tf.int64)
                result = tfa.image.interpolate_spline(
                    slice_coords, reshape_img, chunk_coords, order=3
                )
                result = result[:, :, 0]
                x, y, z = tf.meshgrid(
                    tf.range(chunk_index[1], chunk_index[1] + chunk_shape[1]),
                    tf.range(chunk_index[2], chunk_index[2] + chunk_shape[2]),
                    tf.range(chunk_index[3], chunk_index[3] + chunk_shape[3]),
                    indexing="ij",
                )
                x, y, z = (
                    tf.reshape(x, (-1, 1)),
                    tf.reshape(y, (-1, 1)),
                    tf.reshape(z, (-1, 1)),
                )
                xyz = tf.concat([x, y, z], axis=1)
                map_coords = xyz[
                    tf.newaxis,
                ]
                chunk_index = tf.tensor_scatter_nd_add(
                    chunk_index, [[1]], [chunk_shape[1]]
                )
                # chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[0]], [0])
                if i == chunk_size - 2:
                    chunk_shape = tf.tensor_scatter_nd_update(
                        chunk_shape,
                        [[1]],
                        [
                            tf.shape(coords, out_type=tf.int64)[1]
                            - chunk_index[1]
                        ],
                    )
                total_result = tf.tensor_scatter_nd_add(
                    total_result, map_coords, result
                )
                i = i + 1
                return i, chunk_index, chunk_shape, total_result

            i = tf.constant(0)
            _, chunk_index, chunk_shape, total_result = tf.while_loop(
                cond_to_loop_i,
                body_fn_i,
                [i, chunk_index, chunk_shape, total_result],
            )
            chunk_index = tf.tensor_scatter_nd_add(
                chunk_index, [[2]], [chunk_shape[2]]
            )
            chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[1]], [0])
            chunk_shape = tf.tensor_scatter_nd_update(
                chunk_shape,
                [[1]],
                [tf.shape(coords, out_type=tf.int64)[1] // chunk_size],
            )
            if j == chunk_size - 2:
                chunk_shape = tf.tensor_scatter_nd_update(
                    chunk_shape,
                    [[2]],
                    [tf.shape(coords, out_type=tf.int64)[2] - chunk_index[2]],
                )
            j = j + 1
            return j, chunk_index, chunk_shape, total_result

        j = tf.constant(0)
        _, chunk_index, chunk_shape, total_result = tf.while_loop(
            cond_to_loop_j,
            body_fn_j,
            [j, chunk_index, chunk_shape, total_result],
        )
        chunk_index = tf.tensor_scatter_nd_add(
            chunk_index, [[3]], [chunk_shape[3]]
        )
        chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[2]], [0])
        chunk_shape = tf.tensor_scatter_nd_update(
            chunk_shape, [[2]], [tf.shape(coords)[2] // chunk_size]
        )
        if k == chunk_size - 2:
            chunk_shape = tf.tensor_scatter_nd_update(
                chunk_shape,
                [[3]],
                [tf.shape(coords, out_type=tf.int64)[3] - chunk_index[3]],
            )
        k = k + 1
        return k, chunk_index, chunk_shape, total_result

    k = tf.constant(0)
    _, chunk_index, chunk_shape, total_result = tf.while_loop(
        cond_to_loop_k, body_fn_k, [k, chunk_index, chunk_shape, total_result]
    )
    return total_result


@tf.function
def map_chunk_random_coordinates_3d(img, coords, order=3):
    new_coords = coords[
        tf.newaxis,
    ]
    new_coords = tf.cast(new_coords, tf.float32)
    x, y, z = tf.meshgrid(
        tf.range(tf.shape(img)[0]),
        tf.range(tf.shape(img)[1]),
        tf.range(tf.shape(img)[2]),
        indexing="ij",
    )
    x, y, z = (
        tf.reshape(x, (-1, 1)),
        tf.reshape(y, (-1, 1)),
        tf.reshape(z, (-1, 1)),
    )
    original_coords = tf.concat([x, y, z], axis=1)
    original_coords = original_coords[
        tf.newaxis,
    ]
    original_coords = tf.cast(original_coords, tf.float32)
    random_coordinates_len = tf.shape(new_coords)[1]
    original_coordinates_len = tf.shape(original_coords)[1]
    coords_len = tf.minimum(random_coordinates_len, original_coordinates_len)
    random_indexes = random_sample(tf.range(original_coordinates_len), 0, coords_len)
    random_indexes = tf.sort(random_indexes)
    random_coords = tf.gather(original_coords, random_indexes, axis=1)
    tmp_img = tf.reshape(img, (1, -1, 1))
    tmp_img = tf.cast(tmp_img, tf.float32)
    random_img = tf.gather(tmp_img, random_indexes, axis=1)
    result = tfa.image.interpolate_spline(
        random_coords, random_img, new_coords, order=order
    )
    return result

@tf.function
def map_random_chunk_coordinates_3d(img, coords, order=3, chunk_size=2):
    img = tf.cond(tf.equal(tf.rank(img), tf.constant(3)), lambda: img, lambda: tf.zeros((8, 8, 8)))
    chunk_shape = tf.shape(coords)[1:] // chunk_size
    chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
    chunk_shape = tf.cast(chunk_shape, tf.int64)
    chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
    total_result = tf.zeros(tf.shape(coords)[1:])
    cond_to_loop_k = lambda k, chunk_index, chunk_shape, total_result: tf.less(
        k, tf.constant(chunk_size)
    )

    def body_fn_k(k, chunk_index, chunk_shape, total_result):
        cond_to_loop_j = (
            lambda j, chunk_index, chunk_shape, total_result: tf.less(
                j, tf.constant(chunk_size)
            )
        )

        def body_fn_j(j, chunk_index, chunk_shape, total_result):
            cond_to_loop_i = (
                lambda i, chunk_index, chunk_shape, total_result: tf.less(
                    i, chunk_size
                )
            )

            def body_fn_i(i, chunk_index, chunk_shape, total_result):
                chunk_coords = tf.slice(coords, chunk_index, chunk_shape)
                chunk_coords_0, chunk_coords_1, chunk_coords_2 = (
                    tf.reshape(chunk_coords[0], (-1, 1)),
                    tf.reshape(chunk_coords[1], (-1, 1)),
                    tf.reshape(chunk_coords[2], (-1, 1)),
                )
                chunk_coords = tf.concat(
                    [chunk_coords_0, chunk_coords_1, chunk_coords_2], axis=1
                )
                chunk_min = tf.math.reduce_min(chunk_coords, axis=0)
                chunk_max = tf.math.reduce_max(chunk_coords, axis=0)
                chunk_min, chunk_max = tf.cast(
                    tf.maximum(tf.constant(0.0), tf.floor(chunk_min)), tf.int64
                ), tf.cast(
                    tf.minimum(
                        tf.cast(tf.shape(img), dtype=tf.float32),
                        tf.round(chunk_max),
                    ),
                    tf.int64,
                )
                slice_size = chunk_max - chunk_min
                slice_img = tf.slice(img, chunk_min, slice_size)
                result = map_chunk_random_coordinates_3d(slice_img, chunk_coords, order=3)
                result = result[:, :, 0]
                x, y, z = tf.meshgrid(
                    tf.range(chunk_index[1], chunk_index[1] + chunk_shape[1]),
                    tf.range(chunk_index[2], chunk_index[2] + chunk_shape[2]),
                    tf.range(chunk_index[3], chunk_index[3] + chunk_shape[3]),
                    indexing="ij",
                )
                x, y, z = (
                    tf.reshape(x, (-1, 1)),
                    tf.reshape(y, (-1, 1)),
                    tf.reshape(z, (-1, 1)),
                )
                xyz = tf.concat([x, y, z], axis=1)
                map_coords = xyz[
                    tf.newaxis,
                ]
                chunk_index = tf.tensor_scatter_nd_add(
                    chunk_index, [[1]], [chunk_shape[1]]
                )
                # chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[0]], [0])
                if i == chunk_size - 2:
                    chunk_shape = tf.tensor_scatter_nd_update(
                        chunk_shape,
                        [[1]],
                        [
                            tf.shape(coords, out_type=tf.int64)[1]
                            - chunk_index[1]
                        ],
                    )
                total_result = tf.tensor_scatter_nd_add(
                    total_result, map_coords, result
                )
                i = i + 1
                return i, chunk_index, chunk_shape, total_result

            i = tf.constant(0)
            _, chunk_index, chunk_shape, total_result = tf.while_loop(
                cond_to_loop_i,
                body_fn_i,
                [i, chunk_index, chunk_shape, total_result],
            )
            chunk_index = tf.tensor_scatter_nd_add(
                chunk_index, [[2]], [chunk_shape[2]]
            )
            chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[1]], [0])
            chunk_shape = tf.tensor_scatter_nd_update(
                chunk_shape,
                [[1]],
                [tf.shape(coords, out_type=tf.int64)[1] // chunk_size],
            )
            if j == chunk_size - 2:
                chunk_shape = tf.tensor_scatter_nd_update(
                    chunk_shape,
                    [[2]],
                    [tf.shape(coords, out_type=tf.int64)[2] - chunk_index[2]],
                )
            j = j + 1
            return j, chunk_index, chunk_shape, total_result

        j = tf.constant(0)
        _, chunk_index, chunk_shape, total_result = tf.while_loop(
            cond_to_loop_j,
            body_fn_j,
            [j, chunk_index, chunk_shape, total_result],
        )
        chunk_index = tf.tensor_scatter_nd_add(
            chunk_index, [[3]], [chunk_shape[3]]
        )
        chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[2]], [0])
        chunk_shape = tf.tensor_scatter_nd_update(
            chunk_shape, [[2]], [tf.shape(coords)[2] // chunk_size]
        )
        if k == chunk_size - 2:
            chunk_shape = tf.tensor_scatter_nd_update(
                chunk_shape,
                [[3]],
                [tf.shape(coords, out_type=tf.int64)[3] - chunk_index[3]],
            )
        k = k + 1
        return k, chunk_index, chunk_shape, total_result

    k = tf.constant(0)
    _, chunk_index, chunk_shape, total_result = tf.while_loop(
        cond_to_loop_k, body_fn_k, [k, chunk_index, chunk_shape, total_result]
    )
    return total_result

@tf.function
def map_chunk_coordinates_2d(img, coords, order=3, chunk_size=4):
    img = tf.cond(
        tf.equal(tf.rank(img), tf.constant(2)), lambda: img, lambda: img[0]
    )
    chunk_size = tf.constant(chunk_size, dtype=tf.int64)
    chunk_shape = tf.shape(coords, out_type=tf.int64)[1:] // chunk_size
    chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
    chunk_shape = tf.cast(chunk_shape, tf.int64)
    chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
    total_result = tf.zeros(tf.shape(coords)[1:])
    cond_to_loop_j = lambda j, chunk_index, chunk_shape, total_result: tf.less(
        j, chunk_size
    )

    def body_fn_j(j, chunk_index, chunk_shape, total_result):
        cond_to_loop_i = (
            lambda i, chunk_index, chunk_shape, total_result: tf.less(
                i, chunk_size
            )
        )

        def body_fn_i(i, chunk_index, chunk_shape, total_result):
            chunk_coords = tf.slice(coords, chunk_index, chunk_shape)
            chunk_coords_0, chunk_coords_1, chunk_coords_2 = (
                tf.reshape(chunk_coords[0], (-1, 1)),
                tf.reshape(chunk_coords[1], (-1, 1)),
                tf.reshape(chunk_coords[2], (-1, 1)),
            )
            chunk_coords = tf.concat(
                [chunk_coords_0, chunk_coords_1, chunk_coords_2], axis=1
            )
            chunk_min = tf.math.reduce_min(chunk_coords, axis=0)
            chunk_max = tf.math.reduce_max(chunk_coords, axis=0)
            chunk_min, chunk_max = tf.cast(
                tf.maximum(tf.constant(0.0), tf.floor(chunk_min)), tf.int64
            ), tf.cast(tf.round(chunk_max), tf.int64)
            slice_size = chunk_max - chunk_min + 1
            slice_img = tf.slice(img, chunk_min, slice_size)
            reshape_img = tf.reshape(slice_img, (1, -1, 1))
            slice_x, slice_y, slice_z = tf.meshgrid(
                tf.range(chunk_min[0], chunk_max[0] + 1),
                tf.range(chunk_min[1], chunk_max[1] + 1),
                tf.range(chunk_min[2], chunk_max[2] + 1),
                indexing="ij",
            )
            slice_x = tf.reshape(slice_x, (-1, 1))
            slice_y = tf.reshape(slice_y, (-1, 1))
            slice_z = tf.reshape(slice_z, (-1, 1))
            slice_coords = tf.concat([slice_x, slice_y, slice_z], axis=1)
            slice_coords = tf.cast(
                slice_coords[
                    tf.newaxis,
                ],
                tf.float32,
            )
            chunk_coords = tf.cast(
                chunk_coords[
                    tf.newaxis,
                ],
                tf.float32,
            )
            result = tfa.image.interpolate_spline(
                slice_coords, reshape_img, chunk_coords, order=3
            )
            result = result[:, :, 0]
            x, y, z = tf.meshgrid(
                tf.range(chunk_index[1], chunk_index[1] + chunk_shape[1]),
                tf.range(chunk_index[2], chunk_index[2] + chunk_shape[2]),
                tf.range(chunk_index[3], chunk_index[3] + chunk_shape[3]),
                indexing="ij",
            )
            x, y, z = (
                tf.reshape(x, (-1, 1)),
                tf.reshape(y, (-1, 1)),
                tf.reshape(z, (-1, 1)),
            )
            xyz = tf.concat([x, y, z], axis=1)
            map_coords = xyz[
                tf.newaxis,
            ]
            chunk_index = tf.tensor_scatter_nd_add(
                chunk_index, [[1]], [chunk_shape[1]]
            )
            # chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[0]], [0])
            if i == chunk_size - 2:
                chunk_shape = tf.tensor_scatter_nd_update(
                    chunk_shape,
                    [[1]],
                    [tf.shape(coords, out_type=tf.int64)[1] - chunk_index[1]],
                )
            total_result = tf.tensor_scatter_nd_add(
                total_result, map_coords, result
            )
            i = i + 1
            return i, chunk_index, chunk_shape, total_result

        i = tf.constant(0, dtype=tf.int64)
        _, chunk_index, chunk_shape, total_result = tf.while_loop(
            cond_to_loop_i,
            body_fn_i,
            [i, chunk_index, chunk_shape, total_result],
        )
        chunk_index = tf.tensor_scatter_nd_add(
            chunk_index, [[2]], [chunk_shape[2]]
        )
        chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[1]], [0])
        chunk_shape = tf.tensor_scatter_nd_update(
            chunk_shape,
            [[1]],
            [tf.shape(coords, out_type=tf.int64)[1] // chunk_size],
        )
        if j == chunk_size - 2:
            chunk_shape = tf.tensor_scatter_nd_update(
                chunk_shape,
                [[2]],
                [tf.shape(coords, out_type=tf.int64)[2] - chunk_index[2]],
            )
        j = j + 1
        return j, chunk_index, chunk_shape, total_result

    j = tf.constant(0, dtype=tf.int64)
    _, chunk_index, chunk_shape, total_result = tf.while_loop(
        cond_to_loop_j, body_fn_j, [j, chunk_index, chunk_shape, total_result]
    )

    return total_result


@tf.function
def map_coordinates_2d(img, coords, order=3):
    img = tf.cond(
        tf.equal(tf.rank(img), tf.constant(2)), lambda: img, lambda: img[0]
    )
    new_coords = tf.concat(
        [tf.reshape(coords[0], (-1, 1)), tf.reshape(coords[1], (-1, 1))],
        axis=1,
    )
    new_coords = new_coords[
        tf.newaxis,
    ]
    new_coords = tf.cast(new_coords, tf.float32)
    x, y = tf.meshgrid(
        tf.range(tf.shape(img)[0]), tf.range(tf.shape(img)[1]), indexing="ij"
    )
    x, y = tf.reshape(x, (-1, 1)), tf.reshape(y, (-1, 1))
    original_coords = tf.concat([x, y], axis=1)
    original_coords = original_coords[
        tf.newaxis,
    ]
    original_coords = tf.cast(original_coords, tf.float32)
    tmp_img = tf.reshape(img, (1, -1, 1))
    tmp_img = tf.cast(tmp_img, tf.float32)
    result = tfa.image.interpolate_spline(
        original_coords, tmp_img, new_coords, order=3
    )
    result = tf.reshape(result, tf.shape(coords)[1:])
    return result


@tf.function
def random_crop_fn(data, seg=nan, crop_size=128, margin=[0, 0, 0]):
    return crop(data, seg, crop_size, margin, "ramdom")


@tf.function
def center_crop_fn(data, crop_size, seg=nan):
    return crop(data, seg, crop_size, 0, "center")


@tf.function(experimental_follow_type_hints=True)
def crop(
    data: tf.Tensor,
    seg: tf.Tensor = nan,
    crop_size: tf.Tensor = 128,
    margins: tf.Tensor = [0, 0, 0],
    crop_type: str = "center",
    pad_mode: str = "constant",
    pad_kwargs: Dict[str, tf.Tensor] = {"constant_values": 0},
    pad_mode_seg: str = "constant",
    pad_kwargs_seg: Dict[str, tf.Tensor] = {"constant_values": 0},
):

    data_shape = tf.shape(data, out_type=tf.int64)
    dim = tf.cast(tf.rank(data), dtype=tf.int64) - 2

    if seg is not nan:
        seg_shape = tf.shape(seg, out_type=tf.int64)
        #  other assertion will not be included here because it doesn't influence the result

    # all assertion is removed because it is unnecessary here
    if not isinstance(crop_size, tf.Tensor):
        crop_size = tf.convert_to_tensor(crop_size)
    margins = tf.constant([0, 0, 0], dtype=tf.int64)

    data_return = tf.zeros(tf.concat([data_shape[:2], crop_size], axis=0))
    if seg is not nan:
        seg_return = tf.zeros(tf.concat([seg_shape[:2], crop_size], axis=0))
    else:
        seg_return = nan
    cond_to_loop = lambda b, data_return, seg_return, data_shape, data, seg_shape, seg, crop_type, crop_size, margins, dim, pad_mode, pad_kwargs, pad_mode_seg, pad_kwargs_seg: tf.less(
        b, data_shape[0]
    )

    b = tf.constant(0, dtype=tf.int64)
    (
        _,
        data_return,
        seg_return,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = tf.while_loop(
        cond_to_loop,
        crop_body_fn,
        [
            b,
            data_return,
            seg_return,
            data_shape,
            data,
            seg_shape,
            seg,
            crop_type,
            crop_size,
            margins,
            dim,
            pad_mode,
            pad_kwargs,
            pad_mode_seg,
            pad_kwargs_seg,
        ],
    )
    return data_return, seg_return


@tf.function
def crop_body_fn(
    b,
    data_result,
    seg_result,
    data_shape,
    data,
    seg_shape,
    seg,
    crop_type,
    crop_size,
    margins,
    dim,
    pad_mode,
    pad_kwargs,
    pad_mode_seg,
    pad_kwargs_seg,
):
    data_shape_here = tf.concat(
        [[data_shape[0]], tf.shape(data[b], out_type=tf.int64)], axis=0
    )
    if not isnan(seg):
        seg_shape_here = tf.concat([[seg_shape[0]], tf.shape(seg[b])], axis=0)
    else:  # TODO here, we will not go here because seg will not be nan in real data
        seg_shape_here = tf.concat(
            [[seg_shape[0]], tf.shape(seg[b])], axis=0
        ) + tf.cast(nan, tf.int32)

    if crop_type == "center":
        lbs = tf.cast(
            get_lbs_for_center_crop(crop_size, data_shape_here), tf.int64
        )
    else:
        lbs = tf.cast(
            get_lbs_for_random_crop(crop_size, data_shape_here, margins),
            tf.int64,
        )

    need_to_pad_lb = tf.map_fn(
        lambda d: tf.abs(tf.minimum(tf.constant(0, dtype=tf.int64), lbs[d])),
        elems=tf.range(dim),
    )
    need_to_pad_ub = tf.map_fn(
        lambda d: tf.abs(
            tf.minimum(
                tf.constant(0, tf.int64),
                data_shape_here[d + 2] - (lbs[d] + crop_size[d]),
            )
        ),
        elems=tf.range(dim),
    )
    need_to_pad = tf.concat(
        [
            tf.reshape(need_to_pad_lb, (-1, 1)),
            tf.reshape(need_to_pad_ub, (-1, 1)),
        ],
        axis=1,
    )
    need_to_pad = tf.concat(
        [tf.constant([[0, 0]], dtype=tf.int64), need_to_pad], axis=0
    )

    ubs = tf.map_fn(
        lambda d: tf.minimum(lbs[d] + crop_size[d], data_shape_here[d + 2]),
        elems=tf.range(dim),
        dtype=tf.int64,
    )
    lbs = tf.map_fn(
        lambda d: tf.maximum(tf.constant(0, tf.int64), lbs[d]),
        elems=tf.range(dim),
        dtype=tf.int64,
    )

    slicer_data_begin = tf.map_fn(
        lambda d: lbs[d], elems=tf.range(dim), dtype=tf.int64
    )
    slicer_data_begin = tf.concat(
        [tf.constant([0], dtype=tf.int64), slicer_data_begin], axis=0
    )

    slicer_data_size = tf.map_fn(
        lambda d: ubs[d] - lbs[d], elems=tf.range(dim), dtype=tf.int64
    )
    slicer_data_size = tf.concat(
        [[data_shape_here[1]], slicer_data_size], axis=0
    )
    data_cropped = tf.slice(data[b], slicer_data_begin, slicer_data_size)

    if seg_result is not nan:
        slicer_seg_begin = tf.map_fn(lambda d: lbs[d], elems=tf.range(dim))
        slicer_seg_begin = tf.concat(
            [tf.constant([0], dtype=tf.int64), slicer_seg_begin], axis=0
        )

        slicer_seg_size = tf.map_fn(
            lambda d: ubs[d] - lbs[d], elems=tf.range(dim), dtype=tf.int64
        )
        slicer_seg_size = tf.concat(
            [[seg_shape_here[1]], slicer_seg_size], axis=0
        )
        seg_cropped = tf.slice(seg[b], slicer_seg_begin, slicer_seg_size)

    data_result_b = tf.cond(
        tf.reduce_any(tf.less(tf.constant(0, dtype=tf.int64), need_to_pad)),
        lambda: pad(data_cropped, need_to_pad, pad_mode, pad_kwargs),
        lambda: data_cropped,
    )
    seg_result_b = tf.cond(
        tf.reduce_any(tf.less(tf.constant(0, dtype=tf.int64), need_to_pad)),
        lambda: pad(seg_cropped, need_to_pad, pad_mode_seg, pad_kwargs_seg),
        lambda: seg_cropped,
    )
    data_result = update_tf_channel(data_result, b, data_result_b)
    seg_result = update_tf_channel(seg_result, b, seg_result_b)

    b = b + 1
    return (
        b,
        data_result,
        seg_result,
        data_shape,
        data,
        seg_shape,
        seg,
        crop_type,
        crop_size,
        margins,
        dim,
        pad_mode,
        pad_kwargs,
        pad_mode_seg,
        pad_kwargs_seg,
    )


@tf.function
def pad(data, need_to_pad, pad_mode, pad_kwargs):
    return tf.pad(
        data,
        need_to_pad,
        mode="constant",  # TODO hard code for pad mode.
        constant_values=tf.cast(pad_kwargs["constant_values"], tf.float32),
    )


@tf.function
def get_lbs_for_center_crop(crop_size, data_shape):
    data_shape = tf.cast(data_shape, tf.int64)
    lbs = tf.map_fn(
        lambda i: (data_shape[i + 2] - crop_size[i]) // 2,
        elems=tf.range(tf.shape(data_shape, out_type=tf.int64)[0] - 2),
    )
    return lbs


@tf.function
def get_lbs_for_random_crop(crop_size, data_shape, margins):
    lbs = tf.map_fn(
        lambda i: tf.cond(
            tf.less(margins[i], data_shape[i + 2] - crop_size[i] - margins[i]),
            lambda: tf.random.uniform(
                [],
                minval=margins[i],
                maxval=data_shape[i + 2] - crop_size[i] - margins[i],
                dtype=tf.int64,
            ),
            lambda: (data_shape[i + 2] - crop_size[i]) // 2,
        ),
        elems=tf.range(tf.shape(data_shape, out_type=tf.int64)[0] - 2),
    )
    return lbs


@tf.function
def create_zero_centered_coordinate_mesh(shape):
    coords = tf.convert_to_tensor(
        tf.meshgrid(
            tf.range(shape[0]),
            tf.range(shape[1]),
            tf.range(shape[2]),
            indexing="ij",
        ),
        dtype=tf.float32,
    )
    shape = tf.cast((shape - 1), dtype=tf.float32) / 2.0
    coords = tf.map_fn(
        lambda d: coords[d] - shape[d], elems=tf.range(3), dtype=tf.float32
    )
    # coords = tf.cast(coords, tf.float32)

    return coords


@tf.function
def random_sample(x, axis, size):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample

@tf.function
def random_choice(a, axis, sample_shape=[nan]):
    """

    :param a: tf.Tensor
    :param axis: int axis to sample along
    :param samples_shape: (optional) shape of samples to produce. if not provided, will sample once.
    :returns: tf.Tensor of shape a.shape[:axis] + samples_shape + a.shape[axis + 1:]
    :rtype:

    Examples:
    >>> a = tf.placeholder(shape=(10, 20, 30), dtype=tf.float32)
    >>> random_choice(a, axis=0)
    <tf.Tensor 'GatherV2:0' shape=(1, 20, 30) dtype=float32>
    >>> random_choice(a, axis=1)
    <tf.Tensor 'GatherV2_1:0' shape=(10, 1, 30) dtype=float32>
    >>> random_choice(a, axis=1, samples_shape=(2, 3))
    <tf.Tensor 'GatherV2_2:0' shape=(10, 2, 3, 30) dtype=float32
    >>> random_choice(a, axis=0, samples_shape=(100,))
    <tf.Tensor 'GatherV2_3:0' shape=(100, 20, 30) dtype=float32>
    """
    if isnan(sample_shape):
        sample_shape = tf.cast([1], tf.int32)
    else:
        sample_shape = tf.cast(sample_shape, tf.int32)
    shape = tf.shape(a)
    dim = shape[axis]
    choice_indices = tf.random.uniform(
        sample_shape, minval=0, maxval=dim, dtype=tf.int32
    )
    samples = tf.gather(a, choice_indices, axis=axis)
    """
    if sample_shape == (1,) or sample_shape == 1:
        return samples[0]
    """
    return samples


@tf.function
def update_tf_channel(data, idx, update_value):
    shape = data.get_shape()
    update_value = update_value[
        tf.newaxis,
    ]
    new_data = tf.concat([data[:idx], update_value, data[idx + 1 :]], axis=0)
    new_data.set_shape(shape)
    return new_data


@tf.function
def cubic_spline_interpolation_3d(data, coords):
    x, y = tf.range(tf.shape(data)[0]), tf.range(tf.shape(coords)[0])
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    # cubic_spline_interpolation_2d(data[0], coords[:, 1:])
    interpolate_2d_x = tf.map_fn(
        lambda i: cubic_spline_interpolation_2d(data[i], coords[:, 1:]),
        elems=tf.range(tf.shape(x)[0]),
        dtype=tf.float32,
    )
    result = tf.map_fn(
        lambda i: cubic_spline_interpolation_1d_2(
            interpolate_2d_x[:, i], coords[i, 0:1]
        ),
        elems=tf.range(tf.shape(y)[0]),
        dtype=tf.float32,
    )
    return result[:, 0]


@tf.function
def cubic_spline_interpolation_2d(data, coords):
    # x, y = tf.meshgrid(tf.range(tf.shape(data)[0]), tf.range(tf.shape(data)[1]), indexing='ij')
    x, y = tf.range(tf.shape(data)[0]), tf.range(tf.shape(coords)[0])
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    # cubic_spline_interpolation_1d_2(data[0], coords[:, 1])
    interpolate_1d_x = tf.map_fn(
        lambda i: cubic_spline_interpolation_1d_2(data[i], coords[:, 1]),
        elems=tf.range(tf.shape(x)[0]),
        dtype=tf.float32,
    )
    # interpolate_1d_x = cubic_spline_interpolation_1d_3(data, coords[:, 1])
    # cubic_spline_interpolation_1d_2(interpolate_1d_x[:, 0], coords[0, 0:1])
    result = tf.map_fn(
        lambda i: cubic_spline_interpolation_1d_2(
            interpolate_1d_x[:, i], coords[i, 0:1]
        ),
        elems=tf.range(tf.shape(y)[0]),
        dtype=tf.float32,
    )
    # result = cubic_spline_interpolation_1d_3(tf.transpose(interpolate_1d_x), coords[:, 1])
    # result = tf.linalg.diag_part(result)
    return result[:, 0]
    # return result


@tf.function
def cubic_spline_interpolation_1d(data, coords):
    x = tf.range(tf.shape(data)[0])
    x = tf.cast(x, tf.float32)
    h = tf.map_fn(
        lambda i: x[i + 1] - x[i],
        elems=tf.range(tf.shape(x)[0] - 1),
        dtype=tf.float32,
    )
    A = tf.map_fn(
        lambda i: h[i], elems=tf.range(tf.shape(x)[0] - 2), dtype=tf.float32
    )
    B = tf.map_fn(
        lambda i: 2 * (h[i] + h[i + 1]),
        elems=tf.range(tf.shape(x)[0] - 2),
        dtype=tf.float32,
    )
    C = tf.map_fn(
        lambda i: h[i + 1],
        elems=tf.range(tf.shape(x)[0] - 2),
        dtype=tf.float32,
    )
    D = tf.map_fn(
        lambda i: 6
        * (
            (data[i + 2] - data[i + 1]) / h[i + 1]
            - (data[i + 1] - data[i]) / h[i]
        ),
        elems=tf.range(tf.shape(x)[0] - 2),
        dtype=tf.float32,
    )
    X = thomas_algorithm(A, B, C, D)
    M = tf.map_fn(
        lambda i: X[i - 1],
        elems=tf.range(1, tf.shape(x)[0] - 1),
        dtype=tf.float32,
    )
    M = tf.concat([tf.constant([0.0]), M, tf.constant([0.0])], axis=0)
    a = tf.map_fn(
        lambda i: data[i], elems=tf.range(tf.shape(x)[0] - 1), dtype=tf.float32
    )
    b = tf.map_fn(
        lambda i: (data[i + 1] - data[i]) / h[i]
        - (2 * h[i] * M[i] + h[i] * M[i + 1]) / 6,
        elems=tf.range(tf.shape(x)[0] - 1),
        dtype=tf.float32,
    )
    c = tf.map_fn(
        lambda i: M[i] / 2,
        elems=tf.range(tf.shape(x)[0] - 1),
        dtype=tf.float32,
    )
    d = tf.map_fn(
        lambda i: (M[i + 1] - M[i]) / (6 * h[i]),
        elems=tf.range(tf.shape(x)[0] - 1),
        dtype=tf.float32,
    )
    splines = tf.cast(tf.floor(coords), tf.int32)
    result = tf.map_fn(
        lambda i: a[splines[i]]
        + b[splines[i]] * (coords[i] - tf.cast(splines[i], tf.float32))
        + c[splines[i]]
        * tf.pow((coords[i] - tf.cast(splines[i], tf.float32)), 2)
        + d[splines[i]]
        * tf.pow((coords[i] - tf.cast(splines[i], tf.float32)), 3),
        elems=tf.range(tf.shape(coords)[0]),
        dtype=tf.float32,
    )
    return result


@tf.function
def cubic_spline_interpolation_1d_2(data, coords):
    x = tf.range(tf.shape(data)[0])
    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[0]
    h = x[1:n] - x[: n - 1]
    A = h[: n - 2]
    B = 2 * (h[: n - 2] + h[1 : n - 1])
    C = h[1 : n - 1]
    D = 6 * (
        (data[2:n] - data[1 : n - 1]) / h[1 : n - 1]
        - (data[1 : n - 1] - data[: n - 2]) / h[: n - 2]
    )
    A = tf.linalg.diag(A)
    B = tf.linalg.diag(B)
    C = tf.linalg.diag(C)
    D = tf.reshape(D, (-1, 1))
    zeros = tf.zeros([n - 2, 1])
    A0 = tf.concat([A, zeros, zeros], axis=1)
    B0 = tf.concat([zeros, B, zeros], axis=1)
    C0 = tf.concat([zeros, zeros, C], axis=1)
    MAT = A0 + B0 + C0
    D = tf.concat([tf.constant([[0.0]]), D, tf.constant([[0.0]])], axis=0)
    zeros = tf.zeros([n])
    one1 = tf.tensor_scatter_nd_update(zeros, [[0]], [1.0])
    one2 = tf.tensor_scatter_nd_update(zeros, [[n - 1]], [1])
    MAT = tf.concat(
        [
            one1[
                tf.newaxis,
            ],
            MAT,
            one2[tf.newaxis],
        ],
        axis=0,
    )
    X = tf.linalg.solve(MAT, D)
    M = X[:, 0]
    """
    X = thomas_algorithm(A, B, C, D)
    M = X[0:n-2]
    M = tf.concat([tf.constant([0.]), M, tf.constant([0.])], axis=0)
    """
    a = data[: n - 1]
    b = (data[1:n] - data[0 : n - 1]) / h[0 : n - 1] - (
        2 * h[0 : n - 1] * M[0 : n - 1] + h[0 : n - 1] * M[1:n]
    ) / 6
    c = M[0 : n - 1] / 2
    d = (M[1:n] - M[0 : n - 1]) / (6 * h[0 : n - 1])
    coeff = tf.transpose(
        tf.concat(
            [a[tf.newaxis], b[tf.newaxis], c[tf.newaxis], d[tf.newaxis]],
            axis=0,
        )
    )
    splines = tf.floor(coords)
    x_ = coords - splines
    splines = tf.cast(splines, tf.int32)
    coeff = tf.gather(coeff, splines, axis=0)
    # result = tf.map_fn(lambda i: a[splines[i]] + b[splines[i]] * (coords[i] - tf.cast(splines[i], tf.float32)) + c[splines[i]] * tf.pow((coords[i] - tf.cast(splines[i], tf.float32)), 2) + d[splines[i]] * tf.pow((coords[i] - tf.cast(splines[i], tf.float32)), 3), elems=tf.range(tf.shape(coords)[0]), dtype=tf.float32)
    result = (
        coeff[:, 0]
        + coeff[:, 1] * x_
        + coeff[:, 2] * x_ * x_
        + coeff[:, 3] * x_ * x_ * x_
    )
    return result


@tf.function
def cubic_spline_interpolation_1d_3(data, coords):
    bat = tf.shape(data)[0]
    x = tf.range(tf.shape(data)[1])
    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[0]
    h = x[1:n] - x[: n - 1]
    A = h[: n - 2]
    B = 2 * (h[: n - 2] + h[1 : n - 1])
    C = h[1 : n - 1]
    D = 6 * (
        (data[:, 2:n] - data[:, 1 : n - 1]) / h[1 : n - 1]
        - (data[:, 1 : n - 1] - data[:, : n - 2]) / h[: n - 2]
    )
    A = tf.linalg.diag(A)
    B = tf.linalg.diag(B)
    C = tf.linalg.diag(C)
    D = tf.reshape(D, (bat, -1, 1))
    zeros = tf.zeros([n - 2, 1])
    A0 = tf.concat([A, zeros, zeros], axis=1)
    B0 = tf.concat([zeros, B, zeros], axis=1)
    C0 = tf.concat([zeros, zeros, C], axis=1)
    MAT = A0 + B0 + C0
    D = tf.concat([tf.zeros((bat, 1, 1)), D, tf.zeros((bat, 1, 1))], axis=1)
    zeros = tf.zeros([n])
    one1 = tf.tensor_scatter_nd_update(zeros, [[0]], [1.0])
    one2 = tf.tensor_scatter_nd_update(zeros, [[n - 1]], [1])
    MAT = tf.concat(
        [
            one1[
                tf.newaxis,
            ],
            MAT,
            one2[tf.newaxis],
        ],
        axis=0,
    )
    eye = tf.eye(n, batch_shape=[bat])
    MAT = tf.matmul(eye, MAT)
    X = tf.linalg.solve(MAT, D)
    M = X[:, :, 0]
    a = data[:, : n - 1]
    b = (data[:, 1:n] - data[:, 0 : n - 1]) / h[0 : n - 1] - (
        2 * h[0 : n - 1] * M[:, 0 : n - 1] + h[0 : n - 1] * M[:, 1:n]
    ) / 6
    c = M[:, 0 : n - 1] / 2
    d = (M[:, 1:n] - M[:, 0 : n - 1]) / (6 * h[0 : n - 1])
    coeff = tf.transpose(
        tf.concat(
            [
                a[:, tf.newaxis],
                b[:, tf.newaxis],
                c[:, tf.newaxis],
                d[:, tf.newaxis],
            ],
            axis=1,
        ),
        (0, 2, 1),
    )
    splines = tf.floor(coords)
    x_ = coords - splines
    splines = tf.cast(splines, tf.int32)
    coeff = tf.gather(coeff, splines, axis=1)
    # result = tf.map_fn(lambda i: a[splines[i]] + b[splines[i]] * (coords[i] - tf.cast(splines[i], tf.float32)) + c[splines[i]] * tf.pow((coords[i] - tf.cast(splines[i], tf.float32)), 2) + d[splines[i]] * tf.pow((coords[i] - tf.cast(splines[i], tf.float32)), 3), elems=tf.range(tf.shape(coords)[0]), dtype=tf.float32)
    result = (
        coeff[:, :, 0]
        + coeff[:, :, 1] * x_
        + coeff[:, :, 2] * x_ * x_
        + coeff[:, :, 3] * x_ * x_ * x_
    )
    return result


@tf.function
def thomas_algorithm(A, B, C, D):
    n = tf.shape(A)[0]
    C = tf.tensor_scatter_nd_update(C, [[0]], [C[0] / B[0]])
    D = tf.tensor_scatter_nd_update(D, [[0]], [D[0] / B[0]])
    X_last = D[n - 1]
    i = n - 2
    X = tf.scan(
        lambda a, i: D[i] - C[i] * a,
        elems=tf.range(0, n - 1, 1),
        initializer=X_last,
    )
    X = tf.concat([X, [X_last]], axis=0)
    return X

@tf.function(experimental_follow_type_hints=True)
def volume_resize(input_data: tf.Tensor, target_shape: tf.Tensor, method: str):
    target_shape = tf.cast(target_shape, tf.int32)
    image = tf.transpose(input_data, perm=[1, 2, 0])
    image = tf.image.resize(image, target_shape[1:], method=method)
    image = tf.transpose(image, perm=[2, 0, 1])
    image = tf.image.resize(image, target_shape[:-1], method=method)
    return image


def main():

    '''
    # chunk spline intp test here
    patch_size = [40, 56, 40]
    patch_center_dist_from_border = [30, 30, 30]
    data = tf.ones([1, 1, 70, 83, 64])
    seg = tf.ones([1, 1, 70, 83, 64])
    data, seg = augment_spatial(
        data, seg, patch_size, patch_center_dist_from_border, random_crop=TFbF
    )
    print(data)
    print(seg)
    print(data.shape)
    print(seg.shape)
    '''

    """
    # cubic spline intep 1d test here
    x = tf.range(50)
    x = tf.cast(x, tf.float32)
    y = tf.math.sin(x)
    y = y[tf.newaxis,]
    z = tf.math.cos(x)
    z = z[tf.newaxis,]
    coords = tf.range(0, 49, 0.1)
    coords = tf.cast(coords, tf.float32)
    tf_start = time.time()
    y = tf.concat([y, z], axis=0)
    pred_y = cubic_spline_interpolation_1d_3(y, coords)
    tf_end = time.time()
    tf_time = tf_end - tf_start
    print(y)
    print(pred_y)
    tf_start2 = time.time()
    pred_y2 = cubic_spline_interpolation_1d_2(y, coords)
    tf_end2 = time.time()
    tf_time2 = tf_end2 - tf_start2
    print(pred_y2)
    y, pred_y = y.numpy(), pred_y.numpy()
    pred_y2 = pred_y2.numpy()
    coords = coords.numpy()
    coords = coords.reshape(1, -1)
    sci_start = time.time()
    sci_pred_y = ndimage.map_coordinates(y, coords, order=3)
    var = np.abs(sci_pred_y - pred_y2)
    sci_end = time.time()
    sci_time = sci_end - sci_start
    coords = coords.squeeze()
    print(sci_pred_y)

    plt.subplot(4, 1, 1)
    plt.plot(x, y)
    plt.subplot(4, 1, 2)
    plt.plot(coords, pred_y)
    plt.subplot(4, 1, 3)
    plt.plot(coords, pred_y2)
    plt.subplot(4, 1, 4)
    plt.plot(coords, sci_pred_y)
    plt.show()

    print(np.abs(sci_pred_y, pred_y))
    print(tf_time, tf_time2, sci_time)
    print(var)
    print(tf_time2 / sci_time)
    """

    """
    # cublic spline intep 2d test here
    a = tf.range(73 * 80)
    a = tf.reshape(a, (73, 80))
    a = tf.cast(a, tf.float32)
    coords = tf.random.uniform([40 * 56, 2], minval=0, maxval=39)
    tf_start = time.time()
    result = cubic_spline_interpolation_2d(a, coords)
    tf_end = time.time()
    tf_time = tf_end - tf_start
    # print(result)
    print(tf_time)
    """

    """
    # cubic spline intep 3d test here
    a = tf.range(73 * 80 * 64)
    a = tf.reshape(a, (73, 80, 64))
    a = tf.cast(a, tf.float32)
    coords = tf.random.uniform([40 * 56 * 40, 3], minval=0, maxval=63)
    tf_start = time.time()
    result = cubic_spline_interpolation_3d(a, coords)
    tf_end = time.time()
    tf_time = tf_end - tf_start
    # print(result)
    print(tf_time)
    """

    """
    # cubic spline intep v3 test here
    # cubic spline intep 1d test here
    x = tf.range(50)
    x = tf.cast(x, tf.float32)
    y = tf.math.sin(x)
    y = y[tf.newaxis,]
    z = tf.math.cos(x)
    z = z[tf.newaxis,]
    coords = tf.range(0, 49, 0.1)
    coords = tf.cast(coords, tf.float32)
    tf_start = time.time()
    y = tf.concat([y, z], axis=0)
    pred_y = cubic_spline_interpolation_1d_3(y, coords)
    tf_end = time.time()
    tf_time = tf_end - tf_start
    print(tf_time)
    """
    
    image = tf.random.uniform((10, 20, 10))
    x, y, z = tf.meshgrid(tf.range(tf.shape(image)[0]), tf.range(tf.shape(image)[1]), tf.range(tf.shape(image)[2]), indexing='ij')
    xyz = tf.stack([x, y, z])
    coords = tf.cast(xyz, tf.float32)
    result = map_linear_coordinates_3d(image, coords)
    print(result.shape)


if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        main()
