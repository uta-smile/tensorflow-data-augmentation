import tensorflow as tf


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    @tf.function
    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(**data_dict)


@tf.function
def convert_3d_to_2d_generator(**data_dict):
    # data_dict_copy = {}
    # shp = data_dict['data'].shape
    # data_dict_copy['data'] = tf.reshape(data_dict['data'], (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    # data_dict_copy['orig_shape_data'] = shp
    # shp = data_dict['seg'].shape
    # data_dict_copy['seg'] = tf.reshape(data_dict['seg'], (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    # data_dict_copy['orig_shape_seg'] = shp
    # return data_dict_copy

    shp = data_dict['data'].shape
    data_dict['data'] = tf.reshape(data_dict['data'], (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = tf.reshape(data_dict['seg'], (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    @tf.function
    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(**data_dict)


@tf.function
def convert_2d_to_3d_generator(**data_dict):
    # data_dict_copy = {}
    # shp = data_dict['orig_shape_data']
    # current_shape = data_dict['data'].shape
    # data_dict_copy['data'] = tf.reshape(data_dict['data'], (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    # shp = data_dict['orig_shape_seg']
    # current_shape_seg = data_dict['seg'].shape
    # data_dict_copy['seg'] = tf.reshape(data_dict['seg'], (shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    # for key in data_dict.keys():
    #     if key not in data_dict_copy.keys():
    #         data_dict_copy[key] = data_dict[key]
    # return data_dict_copy

    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = tf.reshape(data_dict['data'], (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = tf.reshape(data_dict['seg'], (shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


if __name__ == "__main__":
    # tensorflow
    images = tf.random.uniform((8, 40, 376, 376))
    labels = tf.random.uniform((8, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32)
    data_dict = {'data': images, 'seg': labels,
                 'orig_shape_data': (8, 2, 20, 376, 376), 'orig_shape_seg': (8, 1, 20, 376, 376)}
    print(data_dict['data'].shape, data_dict['seg'].shape)  # (8, 40, 376, 376) (8, 20, 376, 376)
    data_dict = Convert2DTo3DTransform()(**data_dict)
    print(data_dict.keys(), data_dict['data'].shape, data_dict['seg'].shape)  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
    
