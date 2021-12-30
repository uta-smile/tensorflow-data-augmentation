import tensorflow as tf

# Others
from tfda.base import DTFT, TFDABase, TFT


class Convert3DTo2DTransform(TFDABase):

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: DTFT):
        return convert_3d_to_2d_generator(data_dict)


@tf.function(experimental_follow_type_hints=True)
def convert_3d_to_2d_generator(data_dict: DTFT) -> DTFT:
    data_dict = data_dict.copy()
    # data_dict_copy = {}
    # shp = data_dict['data'].shape
    # data_dict_copy['data'] = tf.reshape(data_dict['data'], (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    # data_dict_copy['orig_shape_data'] = shp
    # shp = data_dict['seg'].shape
    # data_dict_copy['seg'] = tf.reshape(data_dict['seg'], (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    # data_dict_copy['orig_shape_seg'] = shp
    # return data_dict_copy

    shp = data_dict["data"].shape
    data_dict["data"] = tf.reshape(
        data_dict["data"], (shp[0], shp[1] * shp[2], shp[3], shp[4])
    )
    data_dict["orig_shape_data"] = shp
    shp = data_dict["seg"].shape
    data_dict["seg"] = tf.reshape(
        data_dict["seg"], (shp[0], shp[1] * shp[2], shp[3], shp[4])
    )
    data_dict["orig_shape_seg"] = shp
    return data_dict


class Convert2DTo3DTransform(TFDABase):

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: DTFT) -> DTFT:
        return convert_2d_to_3d_generator(data_dict)


# TODO: ERROR
@tf.function(experimental_follow_type_hints=True)
def convert_2d_to_3d_generator(data_dict: DTFT) -> DTFT:
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

    shp = data_dict["orig_shape_data"]
    current_shape = tf.shape(data_dict["data"])
    data_dict["data"] = tf.reshape(
        data_dict["data"],
        (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]),
    )
    shp = data_dict["orig_shape_seg"]
    current_shape_seg = tf.shape(data_dict["seg"])
    data_dict["seg"] = tf.reshape(
        data_dict["seg"],
        (shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]),
    )
    return data_dict


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
        dct_for_where_it_was_used: TFT,
        mask_idx_in_seg: TFT = 1,
        set_outside_to: TFT = 0,
        data_key: TFT = "data",
        seg_key: TFT = "seg",
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
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: DTFT) -> DTFT :
        """Call the transform."""
        data_dict = data_dict.copy()
        seg = data_dict.get(self.seg_key)
        # if seg is None or seg.shape[1] < self.mask_idx_in_seg:
        #     raise Warning(
        #         "mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist"
        #     )
        data = data_dict.get(self.data_key)
        data_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for b in tf.range(tf.shape(data)[0]):
            mask = seg[b, self.mask_idx_in_seg]
            channel_list = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True
            )
            for c in tf.range(tf.shape(data)[1]):
                if tf.size(self.dct_for_where_it_was_used[c]) > 0:
                    # data[b, c][mask < 0] = self.set_outside_to

                    condition = tf.less(mask, 0)
                    case_true = tf.zeros_like(
                        data[b, c]
                    )  # self.set_outside_to = 0
                    case_false = data[b, c]
                    data_b_c = tf.where(condition, case_true, case_false)

                    channel_list = channel_list.write(c, data_b_c)
            data_b = channel_list.stack()
            data_list = data_list.write(b, data_b)

        data_dict[self.data_key] = data_list.stack()
        return data_dict


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        images = tf.random.uniform((8, 2, 20, 376, 376))
        labels = tf.random.uniform(
            (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32
        )
        data_dict = {"data": images, "seg": labels}
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
        data_dict = Convert3DTo2DTransform()(data_dict)
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)

        images = tf.random.uniform((8, 40, 376, 376))
        labels = tf.random.uniform(
            (8, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32
        )
        data_dict = {
            "data": images,
            "seg": labels,
            "orig_shape_data": (8, 2, 20, 376, 376),
            "orig_shape_seg": (8, 1, 20, 376, 376),
        }
        tf.print(
            data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 40, 376, 376) (8, 20, 376, 376)
        # data_dict = Convert2DTo3DTransform()(data_dict)
        tf.print(
            data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
        )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)

        images = tf.random.uniform((1, 2, 2, 2, 2))
        labels = tf.random.uniform(
            (1, 1, 2, 2, 2), minval=0, maxval=3, dtype=tf.int32
        )
        data_dict = {"data": images, "target": labels}
        tf.print(
            data_dict["data"].shape, data_dict["target"].shape
        )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
        tf.print(data_dict["target"])
        # data_dict = ConvertSegmentationToRegionsTransform(
        #     {"0": (1, 2), "1": (2,)}, "target", "target"
        # )(**data_dict)
        tf.print(
            data_dict["data"].shape, data_dict["target"].shape
        )  # (1, 2, 2, 2, 2) (1, 2, 2, 2, 2)
        tf.print(data_dict["target"])

        images = tf.random.uniform((1, 2, 2, 2, 2))
        labels = (
            tf.random.uniform(
                (1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.int32
            )
            - 1
        )
        data_dict = {"data": images, "seg": labels}
        tf.print(
            data_dict["data"].shape, data_dict["seg"].shape
        )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
        tf.print(data_dict)
        data_dict = MaskTransform(
            tf.constant([[0, 0], [1, 0]]), mask_idx_in_seg=0, set_outside_to=0
        )(data_dict)
        tf.print(
            data_dict["data"].shape, data_dict["seg"].shape
        )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
        tf.print(data_dict)
