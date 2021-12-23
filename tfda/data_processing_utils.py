# @author Wenliang Zhong
# @email wenliang.zhong@uta.edu
# @create date 2021-12-14 18:42:00
# @modify date 2021-12-23 12:33:00
# @desc use for data loader and some processing

# Standard Library
import math
import os
import pickle
from copy import deepcopy

import tensorflow as tf

# Others
import tensorflow_addons as tfa

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": True,
    "elastic_deform_alpha": (0.0, 900.0),
    "elastic_deform_sigma": (9.0, 13.0),
    "p_eldef": 0.2,
    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,
    "do_rotation": True,
    "rotation_x": (-15.0 / 360 * 2.0 * math.pi, 15.0 / 360 * 2.0 * math.pi),
    "rotation_y": (-15.0 / 360 * 2.0 * math.pi, 15.0 / 360 * 2.0 * math.pi),
    "rotation_z": (-15.0 / 360 * 2.0 * math.pi, 15.0 / 360 * 2.0 * math.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1, 2),
    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",
    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
    "num_threads": 12
    if "nnUNet_n_proc_DA" not in os.environ
    else int(os.environ["nnUNet_n_proc_DA"]),
    "num_cached_per_thread": 1,
}


default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0.0, 200.0)
default_2D_augmentation_params["elastic_deform_sigma"] = (9.0, 13.0)
default_2D_augmentation_params["rotation_x"] = (
    -180.0 / 360 * 2.0 * math.pi,
    180.0 / 360 * 2.0 * math.pi,
)
default_2D_augmentation_params["rotation_y"] = (
    -0.0 / 360 * 2.0 * math.pi,
    0.0 / 360 * 2.0 * math.pi,
)
default_2D_augmentation_params["rotation_z"] = (
    -0.0 / 360 * 2.0 * math.pi,
    0.0 / 360 * 2.0 * math.pi,
)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (
    0,
    1,
)  # this can be (0, 1, 2) if dummy_2D=True


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = tf.eye(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords_shape = coords.shape
    coords = tf.matmul(
        tf.transpose(tf.reshape(coords, (len(coords), -1))), rot_matrix
    ).transpose()
    coords = tf.reshape(coords, coords_shape)
    return coords


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = tf.constant(
        [
            [1, 0, 0],
            [0, tf.math.cos(angle), -tf.math.sin(angle)],
            [0, tf.math.sin(angle), tf.math.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation_x

    return tf.matmul(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = tf.constant(
        [
            [tf.math.cos(angle), 0, tf.math.sin(angle)],
            [0, 1, 0],
            [-tf.math.sin(angle), 0, tf.math.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation_y

    return tf.matmul(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = tf.constant(
        [
            [tf.math.cos(angle), -tf.math.sin(angle), 0],
            [tf.math.sin(angle), tf.math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    if matrix is None:
        return rotation_z

    return tf.matmul(matrix, rotation_z)


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords_shape = coords.shape
    coords = tf.matmul(
        tf.transpose(tf.reshape(coords, (len(coords), -1))), rot_matrix
    ).transpose()
    coords = tf.reshape(coords, coords_shape)
    return coords


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = tf.constant(
        [
            [tf.math.cos(angle), -math.sin(angle)],
            [tf.math.sin(angle), tf.math.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation

    return tf.matmul(matrix, rotation)


def get_batch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(tf.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(tf.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(tf.abs(rot_z))
    rot_x = min(90 / 360 * 2.0 * tf.pi, rot_x)
    rot_y = min(90 / 360 * 2.0 * tf.pi, rot_y)
    rot_z = min(90 / 360 * 2.0 * tf.pi, rot_z)
    coords = tf.constant(final_patch_size)
    final_shape = tf.identity(coords)
    if len(coords) == 3:
        final_shape = tf.math.reduce_max(
            tf.concat(
                (tf.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)
            ),
            axis=0,
        )
        final_shape = tf.math.reduce_max(
            tf.concat(
                (tf.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)
            ),
            axis=0,
        )
        final_shape = tf.math.reduce_max(
            tf.concat(
                (tf.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)
            ),
            axis=0,
        )
    elif len(coords) == 2:
        final_shape = tf.math.reduce_max(
            tf.concat((tf.abs(rotate_coords_2d(coords, rot_x)), final_shape)),
            axis=0,
        )
    final_shape /= min(scale_range)
    return final_shape.astype(int)


class DataAugmentor:
    def __init__(self, plans_files, jsn) -> None:
        self.plans_file = plans_files
        self.jsn = jsn
        self.plans = None
        self.threeD = None
        self.do_dummy_2D_aug = None
        self.use_mask_for_norm = None
        self.basic_generator_patch_size = None
        self.patch_size = None
        self.batch_size = None
        self.oversample_foregroung_percent = 0.33
        self.pad_all_sides = None
        self.stage = None
        self.data_aug_param = None
        self.pseud_3d_slices = 1

    def initialize(self, training=True, force_load_plans_file=False):
        if force_load_plans_file or self.plans is None:
            self.load_plans_file()  #  here not sure whether pickle can be used in the tpu
        self.process_plans(self.plans)

        self.setup_DA_params()

    def load_plans_file(self):
        with open(self.plans_file, "rb") as f:
            self.plans = pickle.load(f)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans["plans_per_stage"].keys())) == 1, (
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the "
                "case. Please specify which stage of the cascade must be trained"
            )
            self.stage = list(plans["plans_per_stage"].keys())[0]
        self.plans = plans

        stage_plans = self.plans["plans_per_stage"][self.stage]
        self.batch_size = stage_plans["batch_size"]
        self.patch_size = stage_plans[
            "patch_size"
        ]  # here, in the orginal code, the author convert it to np.array. It may need to change later.
        self.do_dummy_2D_aug = stage_plans["do_dumy_2D_data_aug"]
        self.pad_all_sides = None
        self.use_mask_for_norm = plans["use_mask_for_norm"]
        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError(
                "invalid patch size in plans file: %s" % str(self.patch_size)
            )

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_param = default_3D_augmentation_params
            self.data_aug_param["rotation_x"] = (
                -30.0 / 360 * 2.0 * math.pi,
                30.0 / 360 * 2.0 * math.pi,
            )
            self.data_aug_param["rotation_y"] = (
                -30.0 / 360 * 2.0 * math.pi,
                30.0 / 360 * 2.0 * math.pi,
            )
            self.data_aug_param["rotation_z"] = (
                -30.0 / 360 * 2.0 * math.pi,
                30.0 / 360 * 2.0 * math.pi,
            )
            if self.do_dummy_2D_aug:
                self.data_aug_param["dummy_2D"] = True
                print("Using dummy2d data augmentation")
                self.data_aug_param[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_param[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_param[
                    "rotation_x"
                ] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * math.pi,
                    15.0 / 360 * 2.0 * math.pi,
                )
            self.data_aug_param = default_2D_augmentation_params
        self.data_aug_param[
            "mask_was_used_for_normalization"
        ] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_batch_size(
                self.patch_size[1:],
                self.data_aug_param["rotation_x"],
                self.data_aug_param["rotation_y"],
                self.data_aug_param["rotation_z"],
                self.data_aug_param["scale_range"],
            )
            self.basic_generator_patch_size = tf.constant(
                [self.patch_size[0]] + list(self.basic_generator_patch_size)
            )
            assert (
                self is None
            ), f"do dummpy 2d: {self.basic_generator_patch_size}"

        else:
            self.basic_generator_patch_size = get_batch_size(
                self.patch_size,
                self.data_aug_param["rotation_x"],
                self.data_aug_param["rotation_y"],
                self.data_aug_param["rotation_z"],
                self.data_aug_param["scale_range"],
            )
        self.data_aug_param["scale_range"] = (0.7, 1.4)
        self.data_aug_param["do_elastic"] = False
        self.data_aug_param["selected_seg_channels"] = [0]
        self.data_aug_param[
            "patch_size_for_spatial_transform"
        ] = self.patch_size
        self.data_aug_param["num_cached_per_thread"] = 2

    def formalize_data(
        self,
        case_identifier,
        image,
        label,
        class_locations_bytes,
        class_locations_shape,
        return_type,
    ):
        if tf.random.uniform([]) < self.oversample_foregroung_percent:
            force_fg = True
        else:
            force_fg = False
        class_locations = {}
        for i, class_locations_decode in enumerate(class_locations_bytes):
            class_locations_decode = tf.io.decode_raw(
                class_locations_decode, tf.int64
            )
            class_locations[tf.constant(i + 1, dtype=tf.int64)] = tf.reshape(
                class_locations_decode, (class_locations_shape[i], -1)
            )
        need_to_pad = tf.convert_to_tensor(
            self.basic_generator_patch_size, dtype=tf.int64
        ) - tf.convert_to_tensor(self.patch_size, dtype=tf.int64)
        case_all_data = tf.concat([image, label], axis=0)
        # assert need_to_pad is None, f'{need_to_pad}, {case_all_data.shape}, {self.basic_generator_patch_size}'
        for d in range(3):
            if (
                need_to_pad[d] + case_all_data.shape[d + 1]
                < self.basic_generator_patch_size[d]
            ):
                need_to_pad_d = (
                    self.basic_generator_patch_size[d]
                    - case_all_data.shape[d + 1]
                )
                need_to_pad = update_tf_channel(need_to_pad, d, need_to_pad_d)
                #  need_to_pad[d] = self.basic_generator_patch_size[d] - case_all_data.shape[d+1]
        shape = case_all_data.shape[1:]
        lb_x = -need_to_pad[0] // 2
        ub_x = (
            shape[0]
            + need_to_pad[0] // 2
            + need_to_pad[0] % 2
            - self.basic_generator_patch_size[0]
        )
        lb_y = -need_to_pad[1] // 2
        ub_y = (
            shape[1]
            + need_to_pad[1] // 2
            + need_to_pad[1] % 2
            - self.basic_generator_patch_size[1]
        )
        lb_z = -need_to_pad[2] // 2
        ub_z = (
            shape[2]
            + need_to_pad[2] // 2
            + need_to_pad[2] % 2
            - self.basic_generator_patch_size[2]
        )

        if not force_fg:
            bbox_x_lb = tf.random.uniform(
                [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
            )
            bbox_y_lb = tf.random.uniform(
                [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
            )
            bbox_z_lb = tf.random.uniform(
                [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
            )
        else:
            foreground_classes = tf.constant(
                [i for i in class_locations.keys() if i != 0]
            )
            if len(foreground_classes) == 0:
                selected_class = None
                voxels_of_that_class = None
                print(
                    f"case does not contain any foreground classes {case_identifier}"
                )
            else:
                selected_class = random_choice(tf.range(foreground_classes), 0)
                voxels_of_that_class = class_locations[selected_class]

            if voxels_of_that_class is not None:
                selected_voxel = random_choice(voxels_of_that_class, 0)
                bbox_x_lb = max(
                    lb_x,
                    selected_voxel[0]
                    - self.basic_generator_patch_size[0] // 2,
                )
                bbox_y_lb = max(
                    lb_y,
                    selected_voxel[1]
                    - self.basic_generator_patch_size[1] // 2,
                )
                bbox_z_lb = max(
                    lb_z,
                    selected_voxel[2]
                    - self.basic_generator_patch_size[2] // 2,
                )
            else:
                bbox_x_lb = tf.random.uniform(
                    [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
                )
                bbox_y_lb = tf.random.uniform(
                    [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
                )
                bbox_z_lb = tf.random.uniform(
                    [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
                )

        bbox_x_ub = bbox_x_lb + self.basic_generator_patch_size[0]
        bbox_y_ub = bbox_y_lb + self.basic_generator_patch_size[1]
        bbox_z_ub = bbox_z_lb + self.basic_generator_patch_size[2]

        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape[2], bbox_z_ub)

        # assert case_all_data is None, f'{shape}, {valid_bbox_x_lb}, {valid_bbox_x_ub}, {valid_bbox_y_lb}, {valid_bbox_y_ub}, {valid_bbox_z_lb}, {valid_bbox_z_ub}'

        case_all_data = case_all_data[
            :,
            valid_bbox_x_lb:valid_bbox_x_ub,
            valid_bbox_y_lb:valid_bbox_y_ub,
            valid_bbox_z_lb:valid_bbox_z_ub,
        ]
        """
        # just for test here.
        pad = [[0, 0],
                                            [-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)],
                                            [-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)],
                                            [-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)]]
        assert case_all_data is None, f'{pad}'
        """
        image = tf.pad(
            case_all_data[:-1],
            (
                [0, 0],
                [-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)],
                [-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)],
                [-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)],
            ),
            mode="CONSTANT",
        )
        seg = tf.pad(
            case_all_data[-1:],
            (
                [0, 0],
                [-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)],
                [-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)],
                [-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)],
            ),
            mode="CONSTANT",
            constant_values=-1,
        )

        return image, seg

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(
            self.batch_size * (1 - self.oversample_foregroung_percent)
        )

    def formalize_data_3d(self, data):
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

        images, segs = [], []
        zero = tf.constant(0, dtype=tf.int64)
        for i in range(self.batch_size):
            image = tf.io.decode_raw(image_raw[i], tf.as_dtype(tf.float32))
            label = tf.io.decode_raw(label_raw[i], tf.as_dtype(tf.float32))
            original_image, original_label = tf.reshape(
                image, original_image_size[i]
            ), tf.reshape(label, original_label_size[i])
            image = tf.cast(original_image, dtype=tf.float32)
            label = tf.cast(original_label, dtype=tf.float32)
            # TPU doesn't support tf.int64 well, use tf.int64 directly.
            if label.dtype == tf.int64:
                label = tf.cast(label, dtype=tf.int64)
            class_locations = {}
            for c in range(class_locations_bytes[i].shape[0]):
                class_locations_decode = tf.io.decode_raw(
                    class_locations_bytes[i][c], tf.int64
                )
                class_locations[c + 1] = tf.reshape(
                    class_locations_decode, [class_locations_shape[i][c], -1]
                )
            if self.get_do_oversample(i):
                force_fg = True
            else:
                force_fg = False
            case_all_data = tf.concat([image, label], axis=0)
            self.basic_generator_patch_size = tf.convert_to_tensor(
                self.basic_generator_patch_size, dtype=tf.int64
            )
            self.patch_size = tf.convert_to_tensor(
                self.patch_size, dtype=tf.int64
            )
            need_to_pad = self.basic_generator_patch_size - self.patch_size
            for d in range(3):
                if (
                    need_to_pad[d]
                    + tf.shape(case_all_data, out_type=tf.int64)[d + 1]
                    < self.basic_generator_patch_size[d]
                ):
                    need_to_pad_d = (
                        self.basic_generator_patch_size[d]
                        - tf.shape(case_all_data, out_type=tf.int64)[d + 1]
                    )
                    need_to_pad = update_tf_channel(
                        need_to_pad, d, need_to_pad_d
                    )
            shape = tf.shape(case_all_data, out_type=tf.int64)[1:]
            lb_x = -need_to_pad[0] // 2
            ub_x = (
                shape[0]
                + need_to_pad[0] // 2
                + need_to_pad[0] % 2
                - self.basic_generator_patch_size[0]
            )
            lb_y = -need_to_pad[1] // 2
            ub_y = (
                shape[1]
                + need_to_pad[1] // 2
                + need_to_pad[1] % 2
                - self.basic_generator_patch_size[1]
            )
            lb_z = -need_to_pad[2] // 2
            ub_z = (
                shape[2]
                + need_to_pad[2] // 2
                + need_to_pad[2] % 2
                - self.basic_generator_patch_size[2]
            )

            if not force_fg:
                bbox_x_lb = tf.random.uniform(
                    [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
                )
                bbox_y_lb = tf.random.uniform(
                    [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
                )
                bbox_z_lb = tf.random.uniform(
                    [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
                )
            else:
                foreground_classes = [
                    c for c in class_locations.keys() if c != 0
                ]
                if len(foreground_classes) == 0:
                    selected_class = None
                    voxels_of_that_class = None
                    tf.print(
                        f"case does not contain any foreground classes {case_identifier}"
                    )
                else:
                    voxels_of_that_class = class_locations[1]
                    selected_voxel = voxels_of_that_class[0]

                    rand_val = tf.random.uniform(
                        [],
                        minval=0,
                        maxval=len(class_locations),
                        dtype=tf.int64,
                    )
                    for key in class_locations.keys():
                        if tf.constant(key, dtype=tf.int64) == rand_val:
                            voxels_of_that_class = class_locations[key]
                            selected_voxel = random_choice(
                                voxels_of_that_class, 0
                            )[0]

                if voxels_of_that_class is not None:
                    # selected_voxel = random_choice(voxels_of_that_class, 0)
                    # selected_voxel = voxels_of_that_class[0]
                    bbox_x_lb = tf.maximum(
                        lb_x,
                        selected_voxel[0]
                        - self.basic_generator_patch_size[0] // 2,
                    )
                    bbox_y_lb = tf.maximum(
                        lb_y,
                        selected_voxel[1]
                        - self.basic_generator_patch_size[1] // 2,
                    )
                    bbox_z_lb = tf.maximum(
                        lb_z,
                        selected_voxel[2]
                        - self.basic_generator_patch_size[2] // 2,
                    )
                else:
                    bbox_x_lb = tf.random.uniform(
                        [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
                    )
                    bbox_y_lb = tf.random.uniform(
                        [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
                    )
                    bbox_z_lb = tf.random.uniform(
                        [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
                    )

            bbox_x_ub = bbox_x_lb + self.basic_generator_patch_size[0]
            bbox_y_ub = bbox_y_lb + self.basic_generator_patch_size[1]
            bbox_z_ub = bbox_z_lb + self.basic_generator_patch_size[2]

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
            images.append(img)
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
            segs.append(seg)
        images = tf.stack(images)
        segs = tf.stack(segs)
        # assert data is None, f'{images}'
        data["images"] = images
        data["labels"] = segs
        return data

    def formalize_data_2d(self, data):
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
        original_image_size = tf.cast(data["image/shape"], dtype=tf.int64)
        original_label_size = tf.cast(data["label/shape"], dtype=tf.int64)

        images, segs = [], []
        zero = tf.constant(0, dtype=tf.int64)
        for i in range(self.batch_size):
            image = tf.io.decode_raw(image_raw[i], tf.as_dtype(tf.float32))
            label = tf.io.decode_raw(label_raw[i], tf.as_dtype(tf.float32))
            original_image, original_label = tf.reshape(
                image, original_image_size[i]
            ), tf.reshape(label, original_label_size[i])
            image = tf.cast(original_image, dtype=tf.float32)
            label = tf.cast(original_label, dtype=tf.float32)
            # TPU doesn't support tf.int64 well, use tf.int64 directly.
            if label.dtype == tf.int64:
                label = tf.cast(label, dtype=tf.int64)
            class_locations = {}
            for c in range(class_locations_bytes[i].shape[0]):
                class_locations_decode = tf.io.decode_raw(
                    class_locations_bytes[i][c], tf.int64
                )
                class_locations[c + 1] = tf.reshape(
                    class_locations_decode, [class_locations_shape[i][c], -1]
                )

            if self.get_do_oversample(i):
                force_fg = True
            else:
                force_fg = False

            case_all_data = tf.concat([image, label], axis=0)
            # this is for when there is just a 2d slice in case_all_data (2d support)
            if tf.rank(case_all_data) == 3:
                case_all_data = case_all_data[:, tf.newaxis]

            if not force_fg:
                random_slice = tf.random.uniform(
                    [],
                    minval=0,
                    maxval=tf.shape(case_all_data)[1],
                    dtype=tf.int64,
                )
                selected_class = None
            else:
                foreground_classes = [
                    c for c in class_locations.keys() if c > 0
                ]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = random_choice(
                        tf.range(tf.shape(case_all_data)[1]), 0
                    )[0]
                    tf.print(
                        f"case does not contain any foreground classes {case_identifier}"
                    )
                else:
                    voxels_of_that_class = class_locations[1]
                    rand_val = tf.random.uniform(
                        [],
                        minval=0,
                        maxval=len(class_locations),
                        dtype=tf.int64,
                    )
                    for key in class_locations.keys():
                        if tf.constant(key, dtype=tf.int64) == rand_val:
                            voxels_of_that_class = class_locations[key]
                            valid_slices, _ = tf.unique(
                                voxels_of_that_class[:, 0]
                            )
                            random_slice = random_choice(valid_slices, 0)[0]
                            voxels_of_that_class = voxels_of_that_class[
                                voxels_of_that_class[:, 0] == random_slice
                            ]
                            voxels_of_that_class = voxels_of_that_class[:, 1:]
            if self.pseud_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                mn = random_slice - (self.pseud_3d_slices - 1) // 2
                mx = random_slice + (self.pseud_3d_slices - 1) // 2 + 1
                valid_mn = tf.maximum(mn, zero)
                valid_mx = tf.minimum(mx, tf.shape(case_all_data)[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > zero:
                    shp_for_pad = tf.shape(case_all_data)
                    shp_for_pad_1 = need_to_pad_below
                    shp_for_pad = update_tf_channel(
                        shp_for_pad, 1, shp_for_pad_1
                    )
                    case_all_data = tf.concat(
                        [tf.zeros(shp_for_pad), case_all_data], axis=1
                    )
                if need_to_pad_above > zero:
                    shp_for_pad = tf.shape(case_all_data)
                    shp_for_pad_1 = need_to_pad_above
                    shp_for_pad = update_tf_channel(
                        shp_for_pad, 1, shp_for_pad_1
                    )
                    case_all_data = tf.concat(
                        [case_all_data, tf.zeros(shp_for_pad)], axis=1
                    )
                case_all_data = tf.reshape(
                    case_all_data,
                    (
                        -1,
                        tf.shape(case_all_data)[-2],
                        tf.shape(case_all_data)[-1],
                    ),
                )
                case_all_data = tf.concat(
                    [case_all_data, case_all_seg], axis=0
                )

            assert tf.rank(case_all_data) == 3
            self.basic_generator_patch_size = tf.convert_to_tensor(
                self.basic_generator_patch_size, dtype=tf.int64
            )
            self.patch_size = tf.convert_to_tensor(
                self.patch_size, dtype=tf.int64
            )
            need_to_pad = self.basic_generator_patch_size - self.patch_size
            for d in range(2):
                if (
                    tf.shape(need_to_pad)[d] + tf.shape(case_all_data)[d + 1]
                    < self.basic_generator_patch_size[d]
                ):
                    need_to_pad_d = (
                        self.basic_generator_patch_size[d]
                        - tf.shape(case_all_data)[d + 1]
                    )
                    need_to_pad = update_tf_channel(
                        need_to_pad, d, need_to_pad_d
                    )
            shape = tf.shape(case_all_data)[1:]
            lb_x = -need_to_pad[0] // 2
            ub_x = (
                shape[0]
                + need_to_pad[0] // 2
                + need_to_pad[0] % 2
                - self.basic_generator_patch_size[0]
            )
            lb_y = -need_to_pad[1] // 2
            ub_y = (
                shape[1]
                + need_to_pad[1] // 2
                + need_to_pad[1] % 2
                - self.basic_generator_patch_size[1]
            )

            if not force_fg or selected_class is None:
                bbox_x_lb = tf.random.uniform(
                    [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
                )
                bbox_y_lb = tf.random.uniform(
                    [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
                )
            else:
                selected_voxel = random_choice(voxels_of_that_class, 0)[0]
                bbox_x_lb = tf.maximum(
                    lb_x,
                    selected_voxel[0]
                    - self.basic_generator_patch_size[0] // 2,
                )
                bbox_y_lb = tf.maximum(
                    lb_y,
                    selected_voxel[1]
                    - self.basic_generator_patch_size[1] // 2,
                )

            bbox_x_ub = bbox_x_lb + self.basic_generator_patch_size[0]
            bbox_y_ub = bbox_y_lb + self.basic_generator_patch_size[1]

            valid_bbox_x_lb = tf.maximum(zero, bbox_x_lb)
            valid_bbox_x_ub = tf.minimum(shape[0], bbox_x_ub)
            valid_bbox_y_lb = tf.maximum(zero, bbox_y_lb)
            valid_bbox_y_ub = tf.minimum(shape[1], bbox_y_ub)

            case_all_data = case_all_data[
                :,
                valid_bbox_x_lb:valid_bbox_x_ub,
                valid_bbox_y_lb:valid_bbox_y_ub,
            ]

            case_all_data_donly = tf.pad(
                case_all_data[:-1],
                [
                    [0, 0],
                    [
                        -tf.minimum(zero, bbox_x_lb),
                        tf.maximum(bbox_x_ub - shape[0], zero),
                    ],
                    [
                        -tf.minimum(zero, bbox_y_lb),
                        tf.maximum(bbox_y_ub - shape[1], zero),
                    ],
                ],
            )
            case_all_data_segonly = tf.pad(
                case_all_data[-1:],
                [
                    [0, 0],
                    [
                        -tf.minimum(zero, bbox_x_lb),
                        tf.maximum(bbox_x_ub - shape[0], zero),
                    ],
                    [
                        -tf.minimum(zero, bbox_y_lb),
                        tf.maximum(bbox_y_ub - shape[1], zero),
                    ],
                ],
                constant_values=-1,
            )
            images.append(case_all_data_donly)
            segs.append(case_all_data_segonly)
        data["images"] = tf.stack(images)
        data["labels"] = tf.stack(segs)
        return data


class TestSpatialTransform(tf.keras.layers.Layer):
    def __init__(
        self,
        do_dummy_2D,
        patch_size,
        patch_center_dist_from_border=30,
        do_elastic_deform=True,
        alpha=(0.0, 1000.0),
        sigma=(10.0, 13.0),
        do_rotation=True,
        angle_x=(0, 2 * tf.constant(math.pi)),
        angle_y=(0, 2 * tf.constant(math.pi)),
        angle_z=(0, 2 * tf.constant(math.pi)),
        do_scale=True,
        scale=(0.75, 1.25),
        border_mode_data="nearest",
        border_cval_data=0,
        order_data=3,
        data_key="images",
        label_key="labels",
        border_mode_seg="constant",
        border_cval_seg=0,
        order_seg=0,
        random_crop=True,
        p_el_per_sample=1,
        p_scale_per_sample=1,
        p_rot_per_sample=1,
        independent_scale_for_each_axis=False,
        p_rot_per_axis: float = 1,
        p_independent_scale_per_axis: int = 1,
    ) -> None:

        super(TestSpatialTransform, self).__init__(name="TestSpatialTransform")
        self.do_dummy_2D = do_dummy_2D
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.data_key = data_key
        self.seg_key = label_key
        self.p_independent_scale_per_axis = p_independent_scale_per_axis
        self.rng = tf.random.Generator.from_seed(123, alg="philox")

    def __call__(self, data):
        images = data[self.data_key]
        segs = data[self.seg_key]

        if self.patch_size is None:
            patch_size = tf.shape(images)[2:]
        else:
            patch_size = self.patch_size
        ret_val = augment_spatial(
            images,
            segs,
            patch_size=patch_size,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            do_elastic_deform=self.do_elastic_deform,
            alpha=self.alpha,
            sigma=self.sigma,
            do_rotation=self.do_rotation,
            angle_x=self.angle_x,
            angle_y=self.angle_y,
            angle_z=self.angle_z,
            do_scale=self.do_scale,
            scale=self.scale,
            border_mode_data=self.border_mode_data,
            border_cval_data=self.border_cval_data,
            order_data=self.order_data,
            border_mode_seg=self.border_mode_seg,
            border_cval_seg=self.border_cval_seg,
            order_seg=self.order_seg,
            random_crop=self.random_crop,
            p_el_per_sample=self.p_el_per_sample,
            p_scale_per_sample=self.p_scale_per_sample,
            p_rot_per_sample=self.p_rot_per_sample,
            independent_scale_for_each_axis=self.independent_scale_for_each_axis,
            p_rot_per_axis=self.p_rot_per_axis,
            p_independent_scale_per_axis=self.p_independent_scale_per_axis,
        )
        data[self.data_key] = ret_val[0]
        if segs is not None:
            data[self.seg_key] = ret_val[1]

        return data


@tf.function
def augment_spatial(
    data,
    seg,
    patch_size,
    patch_center_dist_from_border=30,
    do_elastic_deform=True,
    alpha=(0.0, 1000.0),
    sigma=(10.0, 13.0),
    do_rotation=True,
    angle_x=(0, 2 * tf.constant(math.pi)),
    angle_y=(0, 2 * tf.constant(math.pi)),
    angle_z=(0, 2 * tf.constant(math.pi)),
    do_scale=True,
    scale=(0.75, 1.25),
    border_mode_data="nearest",
    border_cval_data=0,
    order_data=3,
    border_mode_seg="constant",
    border_cval_seg=0,
    order_seg=3,
    random_crop=True,
    p_el_per_sample=1.0,
    p_scale_per_sample=1.0,
    p_rot_per_sample=1,
    independent_scale_for_each_axis=False,
    p_rot_per_axis: float = 1.0,
    p_independent_scale_per_axis: int = 1,
):
    def augment_per_sample(
        sample_id, patch_size, data, seg, data_result, seg_result
    ):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = True
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
                coords.set_shape([3, None, None, None])
                return d, coords

            _, coords = tf.while_loop(
                loop_cond,
                body_fn,
                [d, coords],
                shape_invariants=[tf.TensorShape(None), coords.get_shape()],
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
            if seg is not None:
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
                        is_seg=True,
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
            if seg is None:
                s = None
            else:
                s = seg[sample_id : sample_id + 1]
            if random_crop:
                # margin = [patch_center_dist_from_border[d] - tf.cast(patch_size[d], dtype=tf.float32) // 2 for d in range(dim)]
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
            if seg is not None:
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
    seg_result = None
    if seg is not None:
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


def interpolate_img(
    img, coords, order=3, mode="nearest", cval=0.0, is_seg=False
):
    unique_labels, _ = tf.unique(tf.reshape(img, (1, -1))[0])
    if is_seg and order != 0:
        # assert img is None, f'{img}'
        result = tf.zeros(tf.shape(coords)[1:], dtype=tf.float32)
        cond_to_loop = lambda img, i, coords, result, order: tf.less(
            i, tf.shape(unique_labels)[0]
        )

        def body_fn(img, i, coords, result, order):
            img, _, coords, result, order = map_coordinates_seg(
                img, unique_labels[i], coords, result, order
            )  # here I force the order = 3
            i = i + 1
            return img, i, coords, result, order

        i = tf.constant(0)
        _, _, _, result, _ = tf.while_loop(
            cond_to_loop, body_fn, [img, i, coords, result, order]
        )
        return result
    else:
        return map_coordinates_img(img, coords, order)


def map_coordinates_seg(seg, cl, coords, result, order):
    seg = tf.cast(tf.equal(seg, cl), dtype=tf.float32)
    # order = tf.cast(order, tf.int64)
    new_seg = tf.cond(
        tf.equal(tf.rank(seg), tf.constant(3)),
        lambda: map_chunk_coordinates_3d(seg, coords, order),
        lambda: map_chunk_coordinates_3d(seg, coords, order),
    )
    indices = tf.where(tf.greater_equal(new_seg, tf.constant(0.5)))
    result = tf.tensor_scatter_nd_update(
        result, indices, tf.ones(tf.shape(indices)[0]) * cl
    )
    return seg, cl, coords, result, order


def map_coordinates_seg_3d(seg, coords, order=3):
    raise NotImplementedError()


def map_coordinates_seg_2d(seg, coords, order=3):
    raise NotImplementedError()


def map_coordinates_img(img, coords, order=3):
    # return tf.cond(tf.equal(tf.rank(img), tf.constant(3)), lambda: map_coordinates_3d(img, coords, order), lambda: map_coordinates_2d(img, coords, order))
    return tf.cond(
        tf.equal(tf.rank(img), tf.constant(3)),
        lambda: map_chunk_coordinates_3d(img, coords, order),
        lambda: map_chunk_coordinates_3d(img, coords, order),
    )


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


def map_chunk_coordinates_3d_tmp(img, coords, order=3, chunk_size=4):
    chunk_shape = tf.shape(coords)[1:] // chunk_size
    chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
    chunk_shape = tf.cast(chunk_shape, tf.int64)
    chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
    total_result = tf.zeros(tf.shape(coords)[1:])
    for k in range(chunk_size):
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
                    slice_coords, reshape_img, chunk_coords, order=order
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
                        [tf.shape(coords)[1] - chunk_index[1]],
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
                chunk_shape, [[1]], [tf.shape(coords)[1] // chunk_size]
            )
            if j == chunk_size - 2:
                chunk_shape = tf.tensor_scatter_nd_update(
                    chunk_shape, [[2]], [tf.shape(coords)[2] - chunk_index[2]]
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
                chunk_shape, [[3]], [tf.shape(coords)[3] - chunk_index[3]]
            )
    return total_result


def map_chunk_coordinates_3d(img, coords, order=3, chunk_size=4):
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
                ), tf.cast(tf.minimum(tf.cast(tf.shape(img), dtype=tf.float32), tf.round(chunk_max)), tf.int64)
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
                        [tf.shape(coords, out_type=tf.int64)[1] - chunk_index[1]],
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
                chunk_shape, [[1]], [tf.shape(coords, out_type=tf.int64)[1] // chunk_size]
            )
            if j == chunk_size - 2:
                chunk_shape = tf.tensor_scatter_nd_update(
                    chunk_shape, [[2]], [tf.shape(coords, out_type=tf.int64)[2] - chunk_index[2]]
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
                chunk_shape, [[3]], [tf.shape(coords, out_type=tf.int64)[3] - chunk_index[3]]
            )
        k = k + 1
        return k, chunk_index, chunk_shape, total_result

    k = tf.constant(0)
    _, chunk_index, chunk_shape, total_result = tf.while_loop(
        cond_to_loop_k, body_fn_k, [k, chunk_index, chunk_shape, total_result]
    )
    return total_result


def map_chunk_coordinates_2d(img, coords, order=3, chunk_size=4):
    chunk_shape = tf.shape(coords)[1:] // chunk_size
    chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
    chunk_shape = tf.cast(chunk_shape, tf.int64)
    chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
    total_result = tf.zeros(tf.shape(coords)[1:])
    cond_to_loop_j = lambda j, chunk_index, chunk_shape, total_result: tf.less(
        j, tf.constant(chunk_size)
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
                slice_coords, reshape_img, chunk_coords, order=order
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
                    chunk_shape, [[1]], [tf.shape(coords)[1] - chunk_index[1]]
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
            chunk_shape, [[1]], [tf.shape(coords)[1] // chunk_size]
        )
        if j == chunk_size - 2:
            chunk_shape = tf.tensor_scatter_nd_update(
                chunk_shape, [[2]], [tf.shape(coords)[2] - chunk_index[2]]
            )
        j = j + 1
        return j, chunk_index, chunk_shape, total_result

    j = tf.constant(0)
    _, chunk_index, chunk_shape, total_result = tf.while_loop(
        cond_to_loop_j, body_fn_j, [j, chunk_index, chunk_shape, total_result]
    )

    return total_result


def map_coordinates_2d(img, coords, order=3):
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
        original_coords, tmp_img, new_coords, order=order
    )
    result = tf.reshape(result, tf.shape(coords)[1:])
    return result


def map_coordinates(
    input,
    coordinates,
    is_seg=False,
    result=None,
    output=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
):
    output_shape = tf.shape(coordinates)[1:]
    output = tf.zeros(output_shape)
    if prefilter and order > 1:
        padded, npad = prepared_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, mode=mode)
    else:
        npad = 0
        filtered = input
    if is_seg:
        pass
    else:
        return output


def geometric_transform():
    pass


def spline_filter(input, order=3, output=tf.float32, mode="mirror"):
    if order not in [0, 1]:
        axis = tf.constant(0)
        cond_to_loop = lambda input, order, axis, mode: tf.less(
            axis, tf.rank(input)
        )
        output, _, _, _ = tf.while_loop(
            cond_to_loop, spline_filter1d, [input, order, axis, mode]
        )
    else:
        output = input
    return output


def spline_filter1d(input, order=3, axis=-1, mode="mirror"):
    pass


def prepared_for_spline_filter(input, mode, cval):
    if mode in ["nearest", "grid-constant"]:
        npad = 12
        if mode == "grid-constant":
            padded = tf.pad(
                input,
                tf.ones((tf.rank(input), 2), dtype=tf.int64) * npad,
                mode="CONSTANT",
                constant_values=cval,
            )
        elif mode == "nearest":
            padded = input
            for _ in range(npad):
                padded = tf.pad(
                    padded,
                    tf.ones((tf.rank(padded), 2), dtype=tf.int64),
                    mode="SYMMETRIC",
                )
    else:
        npad = 0
        padded = input
    return padded, npad


def random_crop_fn(data, seg=None, crop_size=128, margin=[0, 0, 0]):
    return crop(data, seg, crop_size, margin, "ramdom")


def center_crop_fn(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, "center")


def crop(
    data,
    seg=None,
    crop_size=128,
    margins=(0, 0, 0),
    crop_type="center",
    pad_mode="constant",
    pad_kwargs={"constant_values": 0},
    pad_mode_seg="constant",
    pad_kwargs_seg={"constant_values": 0},
):

    data_shape = tf.shape(data, out_type=tf.int64)
    dim = tf.rank(data) - 2

    if seg is not None:
        seg_shape = tf.shape(seg, out_type=tf.int64)
        #  other assertion will not be included here because it doesn't influence the result

    # all assertion is removed because it is unnecessary here
    if not isinstance(crop_size, tf.Tensor):
        crop_size = tf.convert_to_tensor(crop_size)
    if not isinstance(margins, tf.Tensor):
        margins = tf.convert_to_tensor(margins, dtype=tf.int64)

    data_return = tf.zeros(tf.concat([data_shape[:2], crop_size], axis=0))
    if seg is not None:
        seg_return = tf.zeros(tf.concat([seg_shape[:2], crop_size], axis=0))
    else:
        seg_return = None
    cond_to_loop = lambda b, data_result, seg_result: tf.less(b, data_shape[0])

    def body_fn(b, data_result, seg_result):
        data_shape_here = tf.concat(
            [[data_shape[0]], tf.shape(data[b], out_type=tf.int64)], axis=0
        )
        if seg is not None:
            seg_shape_here = tf.concat(
                [[seg_shape[0]], tf.shape(seg[b])], axis=0
            )

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        else:
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)

        need_to_pad_lb = tf.map_fn(
            lambda d: tf.abs(tf.minimum(tf.constant(0, dtype=tf.int64), lbs[d])), elems=tf.range(dim)
        )
        need_to_pad_ub = tf.map_fn(
            lambda d: tf.abs(
                tf.minimum(tf.constant(0, tf.int64), data_shape_here[d + 2] - (lbs[d] + crop_size[d]))
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
        need_to_pad = tf.concat([tf.constant([[0, 0]]), need_to_pad], axis=0)

        ubs = tf.map_fn(
            lambda d: tf.minimum(
                lbs[d] + crop_size[d], data_shape_here[d + 2]
            ),
            elems=tf.range(dim),
        )
        lbs = tf.map_fn(lambda d: tf.maximum(tf.constant(0, tf.int64), lbs[d]), elems=tf.range(dim))

        slicer_data_begin = tf.map_fn(lambda d: lbs[d], elems=tf.range(dim))
        slicer_data_begin = tf.concat(
            [tf.constant([0]), slicer_data_begin], axis=0
        )

        slicer_data_size = tf.map_fn(
            lambda d: ubs[d] - lbs[d], elems=tf.range(dim)
        )
        slicer_data_size = tf.concat(
            [[data_shape_here[1]], slicer_data_size], axis=0
        )
        data_cropped = tf.slice(data[b], slicer_data_begin, slicer_data_size)

        if seg_result is not None:
            slicer_seg_begin = tf.map_fn(lambda d: lbs[d], elems=tf.range(dim))
            slicer_seg_begin = tf.concat(
                [tf.constant([0]), slicer_seg_begin], axis=0
            )

            slicer_seg_size = tf.map_fn(
                lambda d: ubs[d] - lbs[d], elems=tf.range(dim)
            )
            slicer_seg_size = tf.concat(
                [[seg_shape_here[1]], slicer_seg_size], axis=0
            )
            seg_cropped = tf.slice(seg[b], slicer_seg_begin, slicer_seg_size)

        data_result_b = tf.cond(
            tf.reduce_any(tf.less(tf.constant(0), need_to_pad)),
            lambda: pad(data_cropped, need_to_pad, pad_mode, pad_kwargs),
            lambda: data_cropped,
        )
        seg_result_b = tf.cond(
            tf.reduce_any(tf.less(tf.constant(0), need_to_pad)),
            lambda: pad(
                seg_cropped, need_to_pad, pad_mode_seg, pad_kwargs_seg
            ),
            lambda: seg_cropped,
        )
        data_result = update_tf_channel(data_result, b, data_result_b)
        seg_result = update_tf_channel(seg_result, b, seg_result_b)

        b = b + 1
        return b, data_result, seg_result

    b = tf.constant(0, dtype=tf.int64)
    _, data_return, seg_return = tf.while_loop(
        cond_to_loop, body_fn, [b, data_return, seg_return]
    )
    return data_return, seg_return


def pad(data, need_to_pad, pad_mode, pad_kwargs):
    return tf.pad(
        data,
        need_to_pad,
        mode=pad_mode,
        constant_values=pad_kwargs["constant_values"],
    )


def get_lbs_for_center_crop(crop_size, data_shape):
    data_shape = tf.cast(data_shape, tf.int64)
    lbs = tf.map_fn(
        lambda i: (data_shape[i] - crop_size[i]) // 2,
        elems=tf.range(tf.shape(data_shape, out_type=tf.int64)[0] - 2),
    )
    return lbs


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
        elems=tf.range(tf.shape(data_shape)[0] - 2),
    )
    return lbs


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
    """
    for d in range(3):
        coords_d = coords[d] - shape[d]
        coords = update_tf_channel(coords, d, coords_d)
    """
    coords = tf.map_fn(
        lambda d: coords[d] - shape[d], elems=tf.range(3), dtype=tf.float32
    )
    # coords = tf.cast(coords, tf.float32)

    return coords


def random_choice(a, axis, sample_shape=None):
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
    if sample_shape is None:
        sample_shape = [1]
    shape = tf.shape(a)
    dim = shape[axis]
    choice_indices = tf.random.uniform(
        sample_shape, minval=0, maxval=dim, dtype=tf.int64
    )
    samples = tf.gather(a, choice_indices, axis=axis)
    """
    if sample_shape == (1,) or sample_shape == 1:
        return samples[0]
    """
    return samples


def update_tf_channel(data, idx, update_value):
    update_value = update_value[
        tf.newaxis,
    ]
    new_data = tf.concat([data[:idx], update_value, data[idx + 1 :]], axis=0)
    return new_data


def main():
    # patch_size = [10, 20, 10]
    patch_size = [40, 56, 40]
    patch_center_dist_from_border = [30, 30, 30]
    data = tf.ones([1, 1, 70, 83, 64])
    seg = tf.ones([1, 1, 70, 83, 64])
    # data = tf.ones([1, 1, 10, 20, 10])
    # seg = tf.ones([1, 1, 10, 20, 10])
    data, seg = augment_spatial(
        data, seg, patch_size, patch_center_dist_from_border, random_crop=False
    )
    print(data)
    print(seg)
    print(data.shape)
    print(seg.shape)


if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        main()
