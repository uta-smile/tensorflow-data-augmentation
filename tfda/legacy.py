# def formalize_data(
#         self,
#         case_identifier,
#         image,
#         label,
#         class_locations_bytes,
#         class_locations_shape,
#         return_type,
#     ):
#         if tf.random.uniform([]) < self.oversample_foregroung_percent:
#             force_fg = TFbT
#         else:
#             force_fg = TFbF
#         class_locations = {}
#         for i, class_locations_decode in enumerate(class_locations_bytes):
#             class_locations_decode = tf.io.decode_raw(
#                 class_locations_decode, tf.int64
#             )
#             class_locations[tf.constant(i + 1, dtype=tf.int64)] = tf.reshape(
#                 class_locations_decode, (class_locations_shape[i], -1)
#             )
#         need_to_pad = tf.convert_to_tensor(
#             self.basic_generator_patch_size, dtype=tf.int64
#         ) - tf.convert_to_tensor(self.patch_size, dtype=tf.int64)
#         case_all_data = tf.concat([image, label], axis=0)
#         # assert need_to_pad is nan, f'{need_to_pad}, {case_all_data.shape}, {self.basic_generator_patch_size}'
#         def update_need_to_pad(need_to_pad, d):
#             need_to_pad_d = self.basic_generator_patch_size[d] - tf.shape(case_all_data)[d+1]
#             return tf.cond(tf.less(need_to_pad[d]+tf.shape(case_all_data)[d], self.basic_generator_patch_size[d]), lambda: update_tf_channel(need_to_pad, d, need_to_pad_d), lambda: need_to_pad)
#             #  need_to_pad[d] = self.basic_generator_patch_size[d] - case_all_data.shape[d+1]
#         need_to_pad = tf.map_fn(lambda d: update_need_to_pad(need_to_pad, d), elems=tf.range(3, dtype=tf.float32))
#         shape = case_all_data.shape[1:]
#         lb_x = -need_to_pad[0] // 2
#         ub_x = (
#             shape[0]
#             + need_to_pad[0] // 2
#             + need_to_pad[0] % 2
#             - self.basic_generator_patch_size[0]
#         )
#         lb_y = -need_to_pad[1] // 2
#         ub_y = (
#             shape[1]
#             + need_to_pad[1] // 2
#             + need_to_pad[1] % 2
#             - self.basic_generator_patch_size[1]
#         )
#         lb_z = -need_to_pad[2] // 2
#         ub_z = (
#             shape[2]
#             + need_to_pad[2] // 2
#             + need_to_pad[2] % 2
#             - self.basic_generator_patch_size[2]
#         )

#         if not force_fg:
#             bbox_x_lb = tf.random.uniform(
#                 [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
#             )
#             bbox_y_lb = tf.random.uniform(
#                 [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
#             )
#             bbox_z_lb = tf.random.uniform(
#                 [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
#             )
#         else:
#             foreground_classes = tf.constant(
#                 [i for i in class_locations.keys() if i != 0]
#             )
#             if len(foreground_classes) == 0:
#                 selected_class = nan
#                 voxels_of_that_class = nan
#                 print(
#                     f"case does not contain any foreground classes {case_identifier}"
#                 )
#             else:
#                 selected_class = random_choice(tf.range(foreground_classes), 0)
#                 voxels_of_that_class = class_locations[selected_class]

#             if voxels_of_that_class is not nan:
#                 selected_voxel = random_choice(voxels_of_that_class, 0)
#                 bbox_x_lb = max(
#                     lb_x,
#                     selected_voxel[0]
#                     - self.basic_generator_patch_size[0] // 2,
#                 )
#                 bbox_y_lb = max(
#                     lb_y,
#                     selected_voxel[1]
#                     - self.basic_generator_patch_size[1] // 2,
#                 )
#                 bbox_z_lb = max(
#                     lb_z,
#                     selected_voxel[2]
#                     - self.basic_generator_patch_size[2] // 2,
#                 )
#             else:
#                 bbox_x_lb = tf.random.uniform(
#                     [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
#                 )
#                 bbox_y_lb = tf.random.uniform(
#                     [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
#                 )
#                 bbox_z_lb = tf.random.uniform(
#                     [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
#                 )

#         bbox_x_ub = bbox_x_lb + self.basic_generator_patch_size[0]
#         bbox_y_ub = bbox_y_lb + self.basic_generator_patch_size[1]
#         bbox_z_ub = bbox_z_lb + self.basic_generator_patch_size[2]

#         valid_bbox_x_lb = max(0, bbox_x_lb)
#         valid_bbox_x_ub = min(shape[0], bbox_x_ub)
#         valid_bbox_y_lb = max(0, bbox_y_lb)
#         valid_bbox_y_ub = min(shape[1], bbox_y_ub)
#         valid_bbox_z_lb = max(0, bbox_z_lb)
#         valid_bbox_z_ub = min(shape[2], bbox_z_ub)

#         # assert case_all_data is nan, f'{shape}, {valid_bbox_x_lb}, {valid_bbox_x_ub}, {valid_bbox_y_lb}, {valid_bbox_y_ub}, {valid_bbox_z_lb}, {valid_bbox_z_ub}'

#         case_all_data = case_all_data[
#             :,
#             valid_bbox_x_lb:valid_bbox_x_ub,
#             valid_bbox_y_lb:valid_bbox_y_ub,
#             valid_bbox_z_lb:valid_bbox_z_ub,
#         ]
#         """
#         # just for test here.
#         pad = [[0, 0],
#                                             [-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)],
#                                             [-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)],
#                                             [-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)]]
#         assert case_all_data is nan, f'{pad}'
#         """
#         image = tf.pad(
#             case_all_data[:-1],
#             (
#                 [0, 0],
#                 [-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)],
#                 [-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)],
#                 [-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)],
#             ),
#             mode="CONSTANT",
#         )
#         seg = tf.pad(
#             case_all_data[-1:],
#             (
#                 [0, 0],
#                 [-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)],
#                 [-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)],
#                 [-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)],
#             ),
#             mode="CONSTANT",
#             constant_values=-1,
#         )

#         return image, seg

#     def get_do_oversample(self, batch_idx):
#         '''
#         return not batch_idx < tf.round(
#             tf.constant(self.batch_size * (1 - self.oversample_foregroung_percent))
#         )
#         '''
#         return tf.greater_equal(tf.cast(batch_idx, tf.float64), tf.round(
#             self.batch_size * (tf.cast(1, tf.float64) - self.oversample_foregroung_percent))
#         )

'''
def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = tf.eye(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords_shape = tf.shape(coords)
    coords = tf.transpose(
        tf.matmul(
        tf.transpose(tf.reshape(coords, (len(coords), -1))), rot_matrix
    )
    )
    coords = tf.reshape(coords, coords_shape)
    return coords


def create_matrix_rotation_x_3d(angle, matrix=nan):
    rotation_x = tf.convert_to_tensor(
        [
            [1, 0, 0],
            [0, tf.math.cos(angle), -tf.math.sin(angle)],
            [0, tf.math.sin(angle), tf.math.cos(angle)],
        ]
    )
    if matrix is nan:
        return rotation_x

    return tf.matmul(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=nan):
    rotation_y = tf.convert_to_tensor(
        [
            [tf.math.cos(angle), 0, tf.math.sin(angle)],
            [0, 1, 0],
            [-tf.math.sin(angle), 0, tf.math.cos(angle)],
        ]
    )
    if matrix is nan:
        return rotation_y

    return tf.matmul(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=nan):
    rotation_z = tf.convert_to_tensor(
        [
            [tf.math.cos(angle), -tf.math.sin(angle), 0],
            [tf.math.sin(angle), tf.math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    if matrix is nan:
        return rotation_z

    return tf.matmul(matrix, rotation_z)


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords_shape = tf.shape(coords)
    coords = tf.transpose(
        tf.matmul(
        tf.transpose(tf.reshape(coords, (len(coords), -1))), rot_matrix
    )
    )
    coords = tf.reshape(coords, coords_shape)
    return coords


def create_matrix_rotation_2d(angle, matrix=nan):
    rotation = tf.convert_to_tensor(
        [
            [tf.math.cos(angle), -math.sin(angle)],
            [tf.math.sin(angle), tf.math.cos(angle)],
        ]
    )
    if matrix is nan:
        return rotation

    return tf.matmul(matrix, rotation)
'''

'''
class_locations = []
class_locations_len = []
for c in tf.range(tf.shape(class_locations_bytes[i])[0]):
    class_locations_decode = tf.io.decode_raw(
        class_locations_bytes[i][c], tf.int64
    )
    class_locations_c = tf.reshape(
        class_locations_decode, [class_locations_shape[i][c], -1]
    )
    class_locations.append(class_locations_c)
    class_locations_len.append(class_locations_shape[i][c])
class_locations = tf.concat(class_locations, axis=0)
class_locations_len = tf.convert_to_tensor(class_locations_len)
'''

'''
foreground_classes = [
    c for c in class_locations.keys() if c != 0
]
voxels_of_that_class = class_locations[1]
selected_voxel = voxels_of_that_class[0]
if len(foreground_classes) == 0:
    selected_class = nan
    voxels_of_that_class = nan
    tf.print(
        f"case does not contain any foreground classes {case_identifier}"
    )
else:

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
'''

# class TestSpatialTransform(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         do_dummy_2D,
#         patch_size,
#         patch_center_dist_from_border=30,
#         do_elastic_deform=TFbT,
#         alpha=(0.0, 1000.0),
#         sigma=(10.0, 13.0),
#         do_rotation=TFbT,
#         angle_x=(0, 2 * tf.constant(pi)),
#         angle_y=(0, 2 * tf.constant(pi)),
#         angle_z=(0, 2 * tf.constant(pi)),
#         do_scale=TFbT,
#         scale=(0.75, 1.25),
#         border_mode_data="nearest",
#         border_cval_data=0,
#         order_data=3,
#         data_key="images",
#         label_key="labels",
#         border_mode_seg="constant",
#         border_cval_seg=0,
#         order_seg=0,
#         random_crop=TFbT,
#         p_el_per_sample=1,
#         p_scale_per_sample=1,
#         p_rot_per_sample=1,
#         independent_scale_for_each_axis=TFbF,
#         p_rot_per_axis: float = 1,
#         p_independent_scale_per_axis: int = 1,
#     ) -> nan:

#         super(TestSpatialTransform, self).__init__(name="TestSpatialTransform")
#         self.do_dummy_2D = do_dummy_2D
#         self.independent_scale_for_each_axis = independent_scale_for_each_axis
#         self.p_rot_per_sample = p_rot_per_sample
#         self.p_scale_per_sample = p_scale_per_sample
#         self.p_el_per_sample = p_el_per_sample
#         self.patch_size = patch_size
#         self.patch_center_dist_from_border = patch_center_dist_from_border
#         self.do_elastic_deform = do_elastic_deform
#         self.alpha = alpha
#         self.sigma = sigma
#         self.do_rotation = do_rotation
#         self.angle_x = angle_x
#         self.angle_y = angle_y
#         self.angle_z = angle_z
#         self.do_scale = do_scale
#         self.scale = scale
#         self.border_mode_data = border_mode_data
#         self.border_cval_data = border_cval_data
#         self.order_data = order_data
#         self.border_mode_seg = border_mode_seg
#         self.border_cval_seg = border_cval_seg
#         self.order_seg = order_seg
#         self.random_crop = random_crop
#         self.p_rot_per_axis = p_rot_per_axis
#         self.data_key = data_key
#         self.seg_key = label_key
#         self.p_independent_scale_per_axis = p_independent_scale_per_axis
#         self.rng = tf.random.Generator.from_seed(123, alg="philox")

#     def __call__(self, data):
#         images = data[self.data_key]
#         segs = data[self.seg_key]

#         if isnan(self.patch_size):
#             patch_size = tf.shape(images)[2:]
#         else:
#             patch_size = self.patch_size
#         ret_val = augment_spatial(
#             images,
#             segs,
#             patch_size=patch_size,
#             patch_center_dist_from_border=self.patch_center_dist_from_border,
#             do_elastic_deform=self.do_elastic_deform,
#             alpha=self.alpha,
#             sigma=self.sigma,
#             do_rotation=self.do_rotation,
#             angle_x=self.angle_x,
#             angle_y=self.angle_y,
#             angle_z=self.angle_z,
#             do_scale=self.do_scale,
#             scale=self.scale,
#             border_mode_data=self.border_mode_data,
#             border_cval_data=self.border_cval_data,
#             order_data=self.order_data,
#             border_mode_seg=self.border_mode_seg,
#             border_cval_seg=self.border_cval_seg,
#             order_seg=self.order_seg,
#             random_crop=self.random_crop,
#             p_el_per_sample=self.p_el_per_sample,
#             p_scale_per_sample=self.p_scale_per_sample,
#             p_rot_per_sample=self.p_rot_per_sample,
#             independent_scale_for_each_axis=self.independent_scale_for_each_axis,
#             p_rot_per_axis=self.p_rot_per_axis,
#             p_independent_scale_per_axis=self.p_independent_scale_per_axis,
#         )
#         data[self.data_key] = ret_val[0]
#         if segs is not nan:
#             data[self.seg_key] = ret_val[1]

#         return data

# @tf.function
# def augment_spatial(
#     data,
#     seg,
#     patch_size,
#     patch_center_dist_from_border=30,
#     do_elastic_deform=TFbT,
#     alpha=(0.0, 1000.0),
#     sigma=(10.0, 13.0),
#     do_rotation=TFbT,
#     angle_x=(0, 2 * tf.constant(pi)),
#     angle_y=(0, 2 * tf.constant(pi)),
#     angle_z=(0, 2 * tf.constant(pi)),
#     do_scale=TFbT,
#     scale=(0.75, 1.25),
#     border_mode_data="nearest",
#     border_cval_data=0,
#     order_data=3,
#     border_mode_seg="constant",
#     border_cval_seg=0,
#     order_seg=3,
#     random_crop=TFbT,
#     p_el_per_sample=1.0,
#     p_scale_per_sample=1.0,
#     p_rot_per_sample=1,
#     independent_scale_for_each_axis=TFbF,
#     p_rot_per_axis: float = 1.0,
#     p_independent_scale_per_axis: int = 1,
# ):
#     def augment_per_sample(
#         sample_id, patch_size, data, seg, data_result, seg_result
#     ):
#         coords = create_zero_centered_coordinate_mesh(patch_size)
#         modified_coords = TFbT
#         if modified_coords:

#             d = tf.constant(0)
#             loop_cond = lambda d, coords: tf.less(d, dim)

#             def body_fn(d, coords):
#                 if random_crop:
#                     ctr = tf.random.uniform(
#                         [],
#                         patch_center_dist_from_border[d],
#                         tf.cast(tf.shape(data)[d + 2], dtype=tf.float32)
#                         - patch_center_dist_from_border[d],
#                     )
#                 else:
#                     ctr = (
#                         tf.cast(tf.shape(data)[d + 2], dtype=tf.float32) / 2.0
#                         - 0.5
#                     )
#                 coords_d = coords[d] + ctr
#                 coords = update_tf_channel(coords, d, coords_d)
#                 d = d + 1
#                 coords.set_shape([3, nan, nan, nan])
#                 return d, coords

#             _, coords = tf.while_loop(
#                 loop_cond,
#                 body_fn,
#                 [d, coords],
#                 shape_invariants=[tf.TensorShape(nan), coords.get_shape()],
#             )
#             data_sample = tf.zeros(tf.shape(data_result)[1:])
#             channel_id = tf.constant(0)
#             cond_to_loop_data = lambda channel_id, data_sample: tf.less(
#                 channel_id, tf.shape(data)[1]
#             )

#             def body_fn_data(channel_id, data_sample):
#                 data_channel = interpolate_img(
#                     data[sample_id, channel_id],
#                     coords,
#                     order_data,
#                     border_mode_data,
#                     border_cval_data,
#                 )
#                 data_sample = update_tf_channel(
#                     data_sample, channel_id, data_channel
#                 )
#                 channel_id = channel_id + 1
#                 return channel_id, data_sample

#             _, data_sample = tf.while_loop(
#                 cond_to_loop_data, body_fn_data, [channel_id, data_sample]
#             )
#             data_result = update_tf_channel(
#                 data_result, sample_id, data_sample
#             )
#             if seg is not nan:
#                 seg_sample = tf.zeros(tf.shape(seg_result)[1:])
#                 channel_id = tf.constant(0)
#                 cond_to_loop_seg = lambda channel_id, seg_sample: tf.less(
#                     channel_id, tf.shape(seg)[1]
#                 )

#                 def body_fn_seg(channel_id, seg_sample):
#                     seg_channel = interpolate_img(
#                         seg[sample_id, channel_id],
#                         coords,
#                         order_seg,
#                         border_mode_seg,
#                         border_cval_seg,
#                         is_seg=TFbT,
#                     )
#                     seg_sample = update_tf_channel(
#                         seg_sample, channel_id, seg_channel
#                     )
#                     channel_id = channel_id + 1
#                     return channel_id, seg_sample

#                 _, seg_sample = tf.while_loop(
#                     cond_to_loop_seg, body_fn_seg, [channel_id, seg_sample]
#                 )
#                 seg_result = update_tf_channel(
#                     seg_result, sample_id, seg_sample
#                 )
#         else:
#             if isnan(seg):
#                 s = nan
#             else:
#                 s = seg[sample_id : sample_id + 1]
#             if random_crop:
#                 # margin = [patch_center_dist_from_border[d] - tf.cast(patch_size[d], dtype=tf.float32) // 2 for d in tf.range(dim)]
#                 margin = tf.map_fn(
#                     lambda d: tf.cast(
#                         patch_center_dist_from_border[d], dtype=tf.int64
#                     )
#                     - patch_size[d] // 2,
#                     elems=tf.range(dim),
#                 )
#                 d, s = random_crop_fn(
#                     data[sample_id : sample_id + 1], s, patch_size, margin
#                 )
#             else:
#                 d, s = center_crop_fn(
#                     data[sample_id : sample_id + 1], patch_size, s
#                 )
#             data_result = update_tf_channel(data_result, sample_id, d[0])
#             if seg is not nan:
#                 seg_result = update_tf_channel(seg_result, sample_id, s[0])
#         sample_id = sample_id + 1
#         return sample_id, patch_size, data, seg, data_result, seg_result

#     if not isinstance(patch_size, tf.Tensor):
#         patch_size = tf.convert_to_tensor(patch_size)
#     if not isinstance(patch_center_dist_from_border, tf.Tensor):
#         patch_center_dist_from_border = tf.convert_to_tensor(
#             patch_center_dist_from_border, dtype=tf.float32
#         )
#     cond_to_loop = lambda sample_id, patch_size, data, seg, data_result, seg_result: tf.less(
#         sample_id, sample_num
#     )
#     dim = tf.shape(patch_size)[0]
#     seg_result = nan
#     if seg is not nan:
#         seg_result = tf.cond(
#             tf.equal(dim, tf.constant(2)),
#             lambda: tf.zeros(
#                 tf.concat([tf.shape(seg)[:2], patch_size[:2]], axis=0)
#             ),
#             lambda: tf.zeros(
#                 tf.concat([tf.shape(seg)[:2], patch_size[:3]], axis=0)
#             ),
#         )

#     data_result = tf.cond(
#         tf.equal(dim, tf.constant(2)),
#         lambda: tf.zeros(
#             tf.concat([tf.shape(data)[:2], patch_size[:2]], axis=0)
#         ),
#         lambda: tf.zeros(
#             tf.concat([tf.shape(data)[:2], patch_size[:3]], axis=0)
#         ),
#     )
#     sample_num = tf.shape(data)[0]
#     sample_id = tf.constant(0)
#     _, _, _, _, data_result, seg_result = tf.while_loop(
#         cond_to_loop,
#         augment_per_sample,
#         [sample_id, patch_size, data, seg, data_result, seg_result],
#     )
#     return data_result, seg_result

# def map_coordinates_seg_3d(seg, coords, order=3):
#     raise NotImplementedError()


# def map_coordinates_seg_2d(seg, coords, order=3):
#     raise NotImplementedError()

# def map_chunk_coordinates_3d_tmp(img, coords, order=3, chunk_size=4):
#     chunk_shape = tf.shape(coords)[1:] // chunk_size
#     chunk_shape = tf.concat([[tf.shape(coords)[0]], chunk_shape], axis=0)
#     chunk_shape = tf.cast(chunk_shape, tf.int64)
#     chunk_index = tf.zeros(tf.rank(coords), dtype=tf.int64)
#     total_result = tf.zeros(tf.shape(coords)[1:])
#     for k in tf.range(chunk_size):
#         cond_to_loop_j = (
#             lambda j, chunk_index, chunk_shape, total_result: tf.less(
#                 j, tf.constant(chunk_size)
#             )
#         )

#         def body_fn_j(j, chunk_index, chunk_shape, total_result):
#             cond_to_loop_i = (
#                 lambda i, chunk_index, chunk_shape, total_result: tf.less(
#                     i, chunk_size
#                 )
#             )

#             def body_fn_i(i, chunk_index, chunk_shape, total_result):
#                 chunk_coords = tf.slice(coords, chunk_index, chunk_shape)
#                 chunk_coords_0, chunk_coords_1, chunk_coords_2 = (
#                     tf.reshape(chunk_coords[0], (-1, 1)),
#                     tf.reshape(chunk_coords[1], (-1, 1)),
#                     tf.reshape(chunk_coords[2], (-1, 1)),
#                 )
#                 chunk_coords = tf.concat(
#                     [chunk_coords_0, chunk_coords_1, chunk_coords_2], axis=1
#                 )
#                 chunk_min = tf.math.reduce_min(chunk_coords, axis=0)
#                 chunk_max = tf.math.reduce_max(chunk_coords, axis=0)
#                 chunk_min, chunk_max = tf.cast(
#                     tf.maximum(tf.constant(0.0), tf.floor(chunk_min)), tf.int64
#                 ), tf.cast(tf.round(chunk_max), tf.int64)
#                 slice_size = chunk_max - chunk_min + 1
#                 slice_img = tf.slice(img, chunk_min, slice_size)
#                 reshape_img = tf.reshape(slice_img, (1, -1, 1))
#                 slice_x, slice_y, slice_z = tf.meshgrid(
#                     tf.range(chunk_min[0], chunk_max[0] + 1),
#                     tf.range(chunk_min[1], chunk_max[1] + 1),
#                     tf.range(chunk_min[2], chunk_max[2] + 1),
#                     indexing="ij",
#                 )
#                 slice_x = tf.reshape(slice_x, (-1, 1))
#                 slice_y = tf.reshape(slice_y, (-1, 1))
#                 slice_z = tf.reshape(slice_z, (-1, 1))
#                 slice_coords = tf.concat([slice_x, slice_y, slice_z], axis=1)
#                 slice_coords = tf.cast(
#                     slice_coords[
#                         tf.newaxis,
#                     ],
#                     tf.float32,
#                 )
#                 chunk_coords = tf.cast(
#                     chunk_coords[
#                         tf.newaxis,
#                     ],
#                     tf.float32,
#                 )
#                 result = tfa.image.interpolate_spline(
#                     slice_coords, reshape_img, chunk_coords, order=order
#                 )
#                 result = result[:, :, 0]
#                 x, y, z = tf.meshgrid(
#                     tf.range(chunk_index[1], chunk_index[1] + chunk_shape[1]),
#                     tf.range(chunk_index[2], chunk_index[2] + chunk_shape[2]),
#                     tf.range(chunk_index[3], chunk_index[3] + chunk_shape[3]),
#                     indexing="ij",
#                 )
#                 x, y, z = (
#                     tf.reshape(x, (-1, 1)),
#                     tf.reshape(y, (-1, 1)),
#                     tf.reshape(z, (-1, 1)),
#                 )
#                 xyz = tf.concat([x, y, z], axis=1)
#                 map_coords = xyz[
#                     tf.newaxis,
#                 ]
#                 chunk_index = tf.tensor_scatter_nd_add(
#                     chunk_index, [[1]], [chunk_shape[1]]
#                 )
#                 # chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[0]], [0])
#                 if i == chunk_size - 2:
#                     chunk_shape = tf.tensor_scatter_nd_update(
#                         chunk_shape,
#                         [[1]],
#                         [tf.shape(coords)[1] - chunk_index[1]],
#                     )
#                 total_result = tf.tensor_scatter_nd_add(
#                     total_result, map_coords, result
#                 )
#                 i = i + 1
#                 return i, chunk_index, chunk_shape, total_result

#             i = tf.constant(0)
#             _, chunk_index, chunk_shape, total_result = tf.while_loop(
#                 cond_to_loop_i,
#                 body_fn_i,
#                 [i, chunk_index, chunk_shape, total_result],
#             )
#             chunk_index = tf.tensor_scatter_nd_add(
#                 chunk_index, [[2]], [chunk_shape[2]]
#             )
#             chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[1]], [0])
#             chunk_shape = tf.tensor_scatter_nd_update(
#                 chunk_shape, [[1]], [tf.shape(coords)[1] // chunk_size]
#             )
#             if j == chunk_size - 2:
#                 chunk_shape = tf.tensor_scatter_nd_update(
#                     chunk_shape, [[2]], [tf.shape(coords)[2] - chunk_index[2]]
#                 )
#             j = j + 1
#             return j, chunk_index, chunk_shape, total_result

#         j = tf.constant(0)
#         _, chunk_index, chunk_shape, total_result = tf.while_loop(
#             cond_to_loop_j,
#             body_fn_j,
#             [j, chunk_index, chunk_shape, total_result],
#         )
#         chunk_index = tf.tensor_scatter_nd_add(
#             chunk_index, [[3]], [chunk_shape[3]]
#         )
#         chunk_index = tf.tensor_scatter_nd_update(chunk_index, [[2]], [0])
#         chunk_shape = tf.tensor_scatter_nd_update(
#             chunk_shape, [[2]], [tf.shape(coords)[2] // chunk_size]
#         )
#         if k == chunk_size - 2:
#             chunk_shape = tf.tensor_scatter_nd_update(
#                 chunk_shape, [[3]], [tf.shape(coords)[3] - chunk_index[3]]
#             )
#     return total_result

# def map_coordinates(
#     input,
#     coordinates,
#     is_seg=TFbF,
#     result=nan,
#     output=nan,
#     order=3,
#     mode="constant",
#     cval=0.0,
#     prefilter=TFbT,
# ):
#     output_shape = tf.shape(coordinates)[1:]
#     output = tf.zeros(output_shape)
#     if prefilter and order > 1:
#         padded, npad = prepared_for_spline_filter(input, mode, cval)
#         filtered = spline_filter(padded, order, mode=mode)
#     else:
#         npad = 0
#         filtered = input
#     if is_seg:
#         pass
#     else:
#         return output


# def geometric_transform():
#     pass


# def spline_filter(input, order=3, output=tf.float32, mode="mirror"):
#     if order not in [0, 1]:
#         axis = tf.constant(0)
#         cond_to_loop = lambda input, order, axis, mode: tf.less(
#             axis, tf.rank(input)
#         )
#         output, _, _, _ = tf.while_loop(
#             cond_to_loop, spline_filter1d, [input, order, axis, mode]
#         )
#     else:
#         output = input
#     return output


# def spline_filter1d(input, order=3, axis=-1, mode="mirror"):
#     pass


# def prepared_for_spline_filter(input, mode, cval):
#     if mode in ["nearest", "grid-constant"]:
#         npad = 12
#         if mode == "grid-constant":
#             padded = tf.pad(
#                 input,
#                 tf.ones((tf.rank(input), 2), dtype=tf.int64) * npad,
#                 mode="CONSTANT",
#                 constant_values=cval,
#             )
#         elif mode == "nearest":
#             padded = input
#             for _ in tf.range(npad):
#                 padded = tf.pad(
#                     padded,
#                     tf.ones((tf.rank(padded), 2), dtype=tf.int64),
#                     mode="SYMMETRIC",
#                 )
#     else:
#         npad = 0
#         padded = input
#     return padded, npad
