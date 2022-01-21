# Tensorflow
import tensorflow as tf

tf.config.run_functions_eagerly(True)

# Local
from tfda.augmentations.resample_augmentations import (
    augment_linear_downsampling_scipy,
    augment_linear_downsampling_scipy_2D,
)
from tfda.base import TFDABase
from tfda.defs import TFbT, TFDAData, nan
from tfda.utils import isnan, isnotnan



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
                    tf.less(tf.random.uniform(()), self.defs.p_per_sample),
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


class SimulateLowResolutionTransform2D(TFDABase):
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
                    tf.less(tf.random.uniform(()), self.defs.p_per_sample),
                    lambda: augment_linear_downsampling_scipy_2D(
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



if __name__ == "__main__":
    with tf.device("/CPU:0"):
        images = tf.random.uniform((8, 2, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 376, 376), minval=0, maxval=2, dtype=tf.float32
        )
        data_dict = TFDAData(images, labels)
        tf.print(
            data_dict, data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
        # data_dict = SimulateLowResolutionTransform(
        #     zoom_range=(0.5, 1),
        #     per_channel=True,
        #     p_per_channel=0.5,
        #     order_downsample=0,
        #     order_upsample=3,
        #     p_per_sample=0.25,
        #     ignore_axes=(0,),
        # )(data_dict)
        data_dict = SimulateLowResolutionTransform2D(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=nan,
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
