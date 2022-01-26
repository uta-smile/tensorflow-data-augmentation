# Tensorflow
import tensorflow as tf

# Local
from tfda.augmentations.utils import to_one_hot
from tfda.base import TFDABase
from tfda.defs import DTFT, TFDAData


class Convert3DTo2DTransform(TFDABase):
    """Convert 3D to 2D."""

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        dshp = tf.shape(dataset.data)
        data = tf.reshape(
            dataset.data, (dshp[0], dshp[1] * dshp[2], dshp[3], dshp[4])
        )

        sshp = tf.shape(dataset.seg)
        seg = tf.reshape(
            dataset["seg"], (sshp[0], sshp[1] * sshp[2], sshp[3], sshp[4])
        )
        # tf.print('before')

        return TFDAData(data, seg, odshp=dshp, osshp=sshp)


class Convert2DTo3DTransform(TFDABase):
    """Convert 2D to 3D."""

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: TFDAData) -> TFDAData:
        odshp = data_dict.odshp
        # tf.print('after')
        current_shape = tf.shape(data_dict.data)
        reshape_data = tf.reshape(
            data_dict.data,
            (
                odshp[0],
                odshp[1],
                odshp[2],
                current_shape[-2],
                current_shape[-1],
            ),
        )
        osshp = data_dict.osshp
        current_shape_seg = tf.shape(data_dict.seg)
        reshape_seg = tf.reshape(
            data_dict.seg,
            (
                osshp[0],
                osshp[1],
                osshp[2],
                current_shape_seg[-2],
                current_shape_seg[-1],
            ),
        )
        return TFDAData(data=reshape_data, seg=reshape_seg)


class ConvertSegmentationToRegionsTransform(TFDABase):
    def __init__(
        self,
        regions: dict,
        seg_key: str = "seg",
        output_key: str = "seg",
        seg_channel: int = 0,
    ):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: DTFT) -> DTFT:
        data_dict = data_dict.copy()
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = tf.zeros(output_shape, dtype=seg.dtype)

            region_output_list = []
            for b in tf.range(seg_shp[0]):
                region_b_list = []
                for r, k in enumerate(self.regions.keys()):
                    region_b_r = tf.zeros_like(region_output[b, r])
                    for l in self.regions[k]:
                        # region_output[b, r][seg[b, self.seg_channel] == l] = 1

                        condition = tf.equal(seg[b, self.seg_channel], l)
                        region_b_r = tf.where(
                            condition,
                            tf.ones_like(region_output[b, r]),
                            region_b_r,
                        )

                    region_b_list.append(region_b_r)
                region_b = tf.stack(region_b_list)
                region_output_list.append(region_b)
            data_dict[self.output_key] = tf.stack(region_output_list)
        return data_dict


class MaskTransform(TFDABase):
    def __init__(
        self,
        dct_for_where_it_was_used: tf.Tensor,
        mask_idx_in_seg: tf.Tensor = 1,
        set_outside_to: tf.Tensor = 0.0,
        **kws,
    ):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        super().__init__(**kws)
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        seg = dataset.seg
        data = dataset.data

        return dataset.new_data(
            tf.map_fn(
                lambda i: (
                    lambda mask: tf.map_fn(
                        lambda j: tf.cond(
                            tf.greater(
                                tf.size(self.dct_for_where_it_was_used[j]), 0
                            ),
                            lambda: tf.where(
                                tf.less(mask, 0.0),
                                tf.fill(
                                    tf.shape(data[i, j]), self.set_outside_to
                                ),
                                data[i, j],
                            ),
                            lambda: data[i, j],
                        ),
                        tf.range(
                            tf.shape(data)[1],
                        ),
                        fn_output_signature=tf.float32,
                    )
                )(seg[i, self.mask_idx_in_seg]),
                tf.range(tf.shape(data)[0]),
                fn_output_signature=tf.float32,
            )
        )


class OneHotTransform(TFDABase):
    """One hot and transpose transform."""

    def __init__(self, all_seg_labels: tuple, **kws) -> None:
        super().__init__(**kws)
        self._all_seg_labels = all_seg_labels

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        data = tf.transpose(dataset.data, (0, 2, 3, 4, 1))
        seg = to_one_hot(dataset.seg[:, 0], self._all_seg_labels)
        seg = tf.transpose(seg, (0, 2, 3, 4, 1))
        return TFDAData(data, seg)


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        # images = tf.random.uniform((8, 2, 20, 376, 376))
        # labels = tf.random.uniform(
        #     (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32
        # )
        # data_dict = TFDAData(images, labels)
        # tf.print(
        #     data_dict, data_dict["data"].shape, data_dict["seg"].shape
        # )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
        # data_dict = Convert3DTo2DTransform()(data_dict)
        # tf.print(
        #     data_dict, data_dict["data"].shape, data_dict["seg"].shape
        # )  # (8, 40, 376, 376) (8, 20, 376, 376)

        # images = tf.random.uniform((8, 40, 376, 376))
        # labels = tf.random.uniform(
        #     (8, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32
        # )
        # data_dict = {
        #     "data": images,
        #     "seg": labels,
        #     "orig_shape_data": (8, 2, 20, 376, 376),
        #     "orig_shape_seg": (8, 1, 20, 376, 376),
        # }
        # tf.print(
        #     data_dict["data"].shape, data_dict["seg"].shape
        # )  # (8, 40, 376, 376) (8, 20, 376, 376)
        # data_dict = Convert2DTo3DTransform()(data_dict)
        # tf.print(
        #     data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        # )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)

        # images = tf.random.uniform((1, 2, 2, 2, 2))
        # labels = tf.random.uniform(
        #     (1, 1, 2, 2, 2), minval=0, maxval=3, dtype=tf.int32
        # )
        # data_dict = {"data": images, "target": labels}
        # tf.print(
        #     data_dict["data"].shape, data_dict["target"].shape
        # )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
        # tf.print(data_dict["target"])
        # data_dict = ConvertSegmentationToRegionsTransform(
        #     {"0": (1, 2), "1": (2,)}, "target", "target"
        # )(**data_dict)
        # tf.print(
        #     data_dict["data"].shape, data_dict["target"].shape
        # )  # (1, 2, 2, 2, 2) (1, 2, 2, 2, 2)
        # tf.print(data_dict["target"])

        # images = tf.random.uniform((1, 2, 2, 2, 2))
        # labels = (
        #     tf.random.uniform(
        #         (1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.int32
        #     )
        #     - 1
        # )
        # data_dict = TFDAData(images, labels)
        # tf.print(
        #     data_dict["data"].shape, data_dict["seg"].shape
        # )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
        # tf.print(data_dict)
        # mf = MaskTransform(
        #     tf.constant([[0, 0], [1, 0]]), mask_idx_in_seg=0, set_outside_to=0
        # )
        # data_dict = mf(data_dict)
        # tf.print(
        #     data_dict["data"].shape, data_dict["seg"].shape
        # )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
        # tf.print(data_dict.data)

        images = tf.random.uniform([1, 2, 8, 37, 37])
        segs = tf.random.uniform([1, 1, 8, 37, 37])
        data_dict = TFDAData(images, segs)
        data_dict = Convert3DTo2DTransform()(data_dict)
        tf.print(data_dict)
        data_dict = Convert2DTo3DTransform()(data_dict)
