import tensorflow as tf

# Others
from tfda.base import TFDABase
from tfda.defs import TFDAData, TFbT, nan
from tfda.utils import isnan, isnotnan

tf.config.run_functions_eagerly(True)


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

    def __init__(self, channels: tf.Tensor = [nan], **kws) -> None:
        super().__init__(**kws)
        self.channels = tf.convert_to_tensor(channels)

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        return dataset.new_data(
            tf.map_fn(
                lambda xs: tf.cond(
                    tf.random.uniform(()) < self.defs.p_per_sample,
                    lambda: augment_linear_downsampling_scipy(
                        xs,
                        zoom_range=self.defs.zoom_range,
                        per_channel=self.defs.per_channel,
                        p_per_channel=self.defs.p_per_channel,
                        channels=self.channels,
                        order_downsample=self.defs.order_downsample,
                        order_upsample=self.defs.order_upsample,
                        ignore_axes=self.defs.ignore_axes,
                    ),
                    lambda: xs,
                ),
                dataset.data,
            )
        )


@tf.function
def augment_liner_help(
    target_shape: tf.Tensor,
    dim: tf.Tensor,
    shp: tf.Tensor,
    ignore_axes: tf.Tensor,
):
    return tf.map_fn(
        lambda d: tf.cond(
            tf.math.reduce_any(tf.equal(ignore_axes, d)),
            lambda: shp[tf.cast(d, tf.int64)],
            # lambda: target_shape[tf.cast(d, tf.int64)],
            lambda: target_shape[tf.cast(d, tf.int64)],
        ),
        tf.range(dim, dtype=tf.float32),
    )


@tf.function(experimental_follow_type_hints=True)
def augment_linear_downsampling_scipy(
    data_sample: tf.Tensor,
    zoom_range: tf.Tensor = (0.5, 1),
    per_channel: tf.Tensor = TFbT,
    p_per_channel: tf.Tensor = 1.0,
    channels: tf.Tensor = nan,
    order_downsample: tf.Tensor = 1,
    order_upsample: tf.Tensor = 0,
    ignore_axes: tf.Tensor = nan,
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
    shp = tf.shape(data_sample, out_type=tf.int64)[1:]

    target_shape = tf.cast(
        tf.round(
            tf.cast(shp, tf.float32)
            * tf.random.uniform((), zoom_range[0], zoom_range[1])
        ),
        tf.int64,
    )

    channels = tf.cast(
        tf.cond(
            isnan(channels),
            lambda: tf.range(
                tf.shape(data_sample)[0], dtype=tf.float32
            ),  # [0, 1]
            lambda: channels,
        ),
        tf.int64,
    )

    return tf.map_fn(
        lambda c: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: volume_resize(
                volume_resize(
                    data_sample[c],
                    tf.cond(
                        per_channel,
                        lambda: tf.cond(
                            isnotnan(tf.cast(ignore_axes, tf.float32)),
                            lambda: tf.map_fn(
                                lambda i: tf.cond(
                                    tf.math.reduce_any(
                                        tf.equal(
                                            i, tf.cast(ignore_axes, tf.int64)
                                        )
                                    ),
                                    lambda: shp[i],
                                    lambda: target_shape[i],
                                ),
                                tf.range(
                                    tf.shape(target_shape)[0],
                                    dtype=target_shape.dtype,
                                ),
                            ),
                            lambda: tf.cast(
                                tf.round(
                                    tf.cast(shp, tf.float32)
                                    * tf.random.uniform(
                                        (), zoom_range[0], zoom_range[1]
                                    )
                                ),
                                tf.int64,
                            ),
                        ),
                        lambda: target_shape,
                    ),
                    method="nearest",
                ),
                shp,
                method="bicubic",
            ),
            lambda: data_sample[c],
        ),
        channels,
        fn_output_signature=tf.float32,
    )


@tf.function(experimental_follow_type_hints=True)
def augment_linear_downsampling_scipy_v1(
    data_sample: tf.Tensor,
    zoom_range: tf.Tensor = (0.5, 1),
    per_channel: tf.Tensor = TFbT,
    p_per_channel: tf.Tensor = 1.0,
    channels: tf.Tensor = nan,
    order_downsample: tf.Tensor = 1,
    order_upsample: tf.Tensor = 0,
    ignore_axes: tf.Tensor = nan,
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
    shp = tf.shape(data_sample, out_type=tf.int64)[1:]  # [ 20 376 376]
    dim = tf.shape(shp)[0]  # 3

    # NOTE: useless
    # target_shape = tf.cond(
    #     tf.logical_not(per_channel),
    #     lambda: tf.round(shp * tf.random.uniform((), zoom_range[0], zoom_range[1])),
    #     lambda: target_shape
    # )
    target_shape = tf.cast(
        tf.round(
            tf.cast(shp, tf.float32)
            * 1.0  # tf.random.uniform((), zoom_range[0], zoom_range[1])
        ),
        tf.int64,
    )

    channels = tf.cast(
        tf.cond(
            isnan(channels),
            lambda: tf.range(
                tf.shape(data_sample)[0], dtype=tf.float32
            ),  # [0, 1]
            lambda: channels,
        ),
        tf.int64,
    )

    data_sample_c_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    for c in channels:
        if tf.less(tf.random.uniform(()), p_per_channel):
            # zoom = uniform(zoom_range[0], zoom_range[1])
            # zoom = 1.0
            zoom = tf.random.uniform(
                (), minval=zoom_range[0], maxval=zoom_range[1]
            )  # 0.8637516095857263

            target_shape = tf.round(tf.cast(shp, tf.float32) * zoom)
            # target_shape = tf.cast(target_shape, tf.int32)  # [ 17 325 325]

            target_shape_list = tf.TensorArray(
                tf.float32, size=tf.size(target_shape)
            )
            target_shape_list = target_shape_list.unstack(target_shape)
            if isnotnan(tf.cast(ignore_axes, tf.float32)):  # ignore_axes = 0
                for i in tf.range(dim):
                    condition = tf.math.reduce_any(
                        ignore_axes == tf.cast(i, tf.int32)
                    )
                    case_true = shp[i]
                    case_false = target_shape[i]
                    target_shape_i = tf.where(
                        condition,
                        tf.cast(case_true, tf.float32),
                        tf.cast(case_false, tf.float32),
                    )
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
def volume_resize(
    input_data: tf.Tensor, target_shape: tf.Tensor, method: tf.Tensor
):
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
        data_dict = TFDAData(images, labels)
        tf.print(
            data_dict, data_dict["data"].shape, data_dict["seg"].shape
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
            data_dict, data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)

        # NOTE: before test this, change zoom in both function to 1.
        # tf.print(
        #     tf.math.reduce_all(
        #         tf.equal(
        #             augment_linear_downsampling_scipy(
        #                 images[0],
        #                 p_per_channel=1.0,
        #                 order_downsample=0,
        #                 order_upsample=3,
        #                 ignore_axes=(0,),
        #             ),
        #             augment_linear_downsampling_scipy_v1(
        #                 images[0],
        #                 p_per_channel=1.0,
        #                 order_downsample=0,
        #                 order_upsample=3,
        #                 ignore_axes=(0,),
        #             ),
        #         )
        #     )
        # )
