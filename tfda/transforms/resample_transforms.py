import tensorflow as tf

# Others
from tfda.base import TFDABase


class SimulateLowResolutionTransform(TFDABase):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="data", p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, **data_dict):
        data_list = []
        for b in range(len(data_dict[self.data_key])):
            if tf.random.uniform(()) < self.p_per_sample:
                data_b = augment_linear_downsampling_scipy(data_dict[self.data_key][b],
                                                           zoom_range=self.zoom_range,
                                                           per_channel=self.per_channel,
                                                           p_per_channel=self.p_per_channel,
                                                           channels=self.channels,
                                                           order_downsample=self.order_downsample,
                                                           order_upsample=self.order_upsample,
                                                           ignore_axes=self.ignore_axes)
            else:
                data_b = data_dict[self.data_key][b]
            data_list.append(data_b)
        data_dict[self.data_key] = tf.stack(data_list)
        return data_dict


@tf.function
def augment_linear_downsampling_scipy(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel: probability for downsampling/upsampling a channel

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:

        ignore_axes: tuple/list

    '''
    if not isinstance(zoom_range, (list, tuple)):
        zoom_range = [zoom_range]
    # data_sample.shape = [2 20 376 376]
    shp = data_sample.shape[1:]  # [ 20 376 376]
    shp = tf.cast(shp, tf.float32)
    dim = len(shp)  # 3

    if channels is None:
        channels = list(range(data_sample.shape[0]))  # [0, 1]

    data_sample_c_list = []
    for c in channels:
        if tf.random.uniform(()) < p_per_channel:
            # zoom = uniform(zoom_range[0], zoom_range[1])
            zoom = tf.random.uniform((), minval=zoom_range[0], maxval=zoom_range[1])  # 0.8637516095857263

            target_shape = tf.round(shp * zoom)
            # target_shape = tf.cast(target_shape, tf.int32)  # [ 17 325 325]

            if ignore_axes is not None:  # ignore_axes = 0
                target_shape_list = []
                for i in range(dim):
                    condition = i in ignore_axes
                    case_true = shp[i]
                    case_false = target_shape[i]
                    target_shape_i = tf.where(condition, case_true, case_false)
                    target_shape_list.append(target_shape_i)
                target_shape = tf.stack(target_shape_list)

            # downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
            #                      anti_aliasing=False)
            # data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge',
            #                         anti_aliasing=False)
            downsampled = volume_resize(data_sample[c], target_shape, method='nearest')
            data_sample_c = volume_resize(downsampled, shp, method='bicubic')
        else:
            data_sample_c = data_sample[c]
        data_sample_c_list.append(data_sample_c)
    data_sample = tf.stack(data_sample_c_list)
    return data_sample


@tf.function
def volume_resize(input_data, target_shape, method):
    target_shape = tf.cast(target_shape, tf.int32)
    image = tf.transpose(input_data, perm=[1, 2, 0])
    image = tf.image.resize(image, target_shape[1:], method=method)
    image = tf.transpose(image, perm=[2, 0, 1])
    image = tf.image.resize(image, target_shape[:-1], method=method)
    return image




if __name__ == "__main__":
    images = tf.random.uniform((8, 2, 20, 376, 376))
    labels = tf.random.uniform((8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32)
    data_dict = {'data': images, 'seg': labels}
    print(data_dict.keys(), data_dict['data'].shape, data_dict['seg'].shape)  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
    data_dict = SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                               order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                               ignore_axes=(0,))(**data_dict)
    print(data_dict.keys(), data_dict['data'].shape, data_dict['seg'].shape)  # (8, 40, 376, 376) (8, 20, 376, 376)
