import tensorflow as tf

# Others
from tfda.base import DTFT, TFDABase, TFT
from tfda.utils import nan


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

    def __init__(
        self,
        zoom_range: TFT = (0.5, 1),
        per_channel: TFT = False,
        p_per_channel: TFT = 1.0,
        channels: TFT = nan,
        order_downsample: TFT = 1,
        order_upsample: TFT = 0,
        data_key: TFT = "data",
        p_per_sample: TFT = 1.0,
        ignore_axes: TFT = nan,
        **kws
    ):
        super().__init__(**kws)
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def call(self, data_dict: DTFT) -> DTFT:
        """Call the transform."""
        data_dict[self.data_key] = tf.map_fn(
            lambda xs: tf.cond(
                tf.random.uniform(()) < self.p_per_sample,
                lambda: augment_linear_downsampling_scipy(
                    xs,
                    zoom_range=self.zoom_range,
                    per_channel=self.per_channel,
                    p_per_channel=self.p_per_channel,
                    channels=self.channels,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                    ignore_axes=self.ignore_axes,
                ),
                lambda: xs,
            ),
            data_dict[self.data_key],
        )
        return data_dict


@tf.function
def augment_liner_help(target_shape: TFT, dim: TFT, shp: TFT, ignore_axes: TFT):
    return tf.map_fn(
        lambda d: tf.cond(
            tf.math.reduce_any(ignore_axes == d),
            lambda: shp[tf.cast(d, tf.int64)],
            # lambda: target_shape[tf.cast(d, tf.int64)],
            lambda: target_shape[tf.cast(d, tf.int64)],
        ),
        tf.range(dim, dtype=tf.float32),
    )


@tf.function
def augment_linear_downsampling_scipy(
    data_sample: TFT,
    zoom_range: TFT = (0.5, 1),
    per_channel: TFT = True,
    p_per_channel: TFT = 1.0,
    channels: TFT = nan,
    order_downsample: TFT = 1,
    order_upsample: TFT = 0,
    ignore_axes: TFT = nan,
):
    """
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

    """
    # if not isinstance(zoom_range, (list, tuple)):
    #     zoom_range = [zoom_range]
    # data_sample.shape = [2 20 376 376]
    shp = tf.shape(data_sample)[1:]  # [ 20 376 376]
    shp = tf.cast(shp, tf.float32)
    dim = tf.shape(shp)[0]  # 3

    if tf.math.reduce_any(tf.math.is_nan(channels)):
        channels = tf.range(
            tf.shape(data_sample)[0], dtype=tf.float32
        )  # [0, 1]

    data_sample_c_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    for c in channels:
        if tf.random.uniform(()) < p_per_channel:
            # zoom = uniform(zoom_range[0], zoom_range[1])
            zoom = tf.random.uniform(
                (), minval=zoom_range[0], maxval=zoom_range[1]
            )  # 0.8637516095857263

            target_shape = tf.round(shp * zoom)
            # target_shape = tf.cast(target_shape, tf.int32)  # [ 17 325 325]

            target_shape_list = tf.TensorArray(
                tf.float32, size=tf.size(target_shape)
            )
            target_shape_list = target_shape_list.unstack(target_shape)
            if not tf.math.reduce_any(
                tf.math.is_nan(tf.cast(ignore_axes, tf.float32))
            ):  # ignore_axes = 0
                for i in tf.range(dim):
                    condition = tf.math.reduce_any(
                        ignore_axes == tf.cast(i, tf.float32)
                    )
                    case_true = shp[i]
                    case_false = target_shape[i]
                    target_shape_i = tf.where(condition, case_true, case_false)
                    target_shape_list = target_shape_list.write(
                        i, target_shape_i
                    )
                target_shape = target_shape_list.stack()
            downsampled = volume_resize(
                data_sample[tf.cast(c, tf.int64)],
                target_shape,
                method="nearest",
            )
            data_sample_c = volume_resize(downsampled, shp, method="bicubic")
        else:
            data_sample_c = data_sample[tf.cast(c, tf.int64)]
        data_sample_c_list = data_sample_c_list.write(
            tf.cast(c, tf.int32), data_sample_c
        )
    data_sample = data_sample_c_list.stack()
    return data_sample


@tf.function
def volume_resize(input_data: TFT, target_shape: TFT, method: TFT):
    target_shape = tf.cast(target_shape, tf.int32)
    image = tf.transpose(input_data, perm=[1, 2, 0])
    image = tf.image.resize(image, target_shape[1:], method=method)
    image = tf.transpose(image, perm=[2, 0, 1])
    image = tf.image.resize(image, target_shape[:-1], method=method)
    return image


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        images = tf.random.uniform((8, 2, 20, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.float32
        )
        data_dict = {"data": images, "seg": labels}
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
        data_dict = SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=(0,),
        )(data_dict)
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)
