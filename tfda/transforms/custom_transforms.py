import tensorflow as tf

# Others
from tfda.base import TFDABase


class Convert3DTo2DTransform(TFDABase):
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

    shp = data_dict["orig_shape_data"]
    current_shape = data_dict["data"].shape
    data_dict["data"] = tf.reshape(
        data_dict["data"],
        (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]),
    )
    shp = data_dict["orig_shape_seg"]
    current_shape_seg = data_dict["seg"].shape
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

    @tf.function
    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = tf.zeros(output_shape, dtype=seg.dtype)

            region_output_list = []
            for b in range(seg_shp[0]):
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
        dct_for_where_it_was_used,
        mask_idx_in_seg=1,
        set_outside_to=0,
        data_key="data",
        seg_key="seg",
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
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    @tf.function
    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning(
                "mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist"
            )
        data = data_dict.get(self.data_key)
        data_list = []
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            channel_list = []
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    # data[b, c][mask < 0] = self.set_outside_to

                    condition = tf.less(mask, 0)
                    case_true = tf.zeros_like(
                        data[b, c]
                    )  # self.set_outside_to = 0
                    case_false = data[b, c]
                    data_b_c = tf.where(condition, case_true, case_false)

                    channel_list.append(data_b_c)
            data_b = tf.stack(channel_list)
            data_list.append(data_b)

        data_dict[self.data_key] = tf.stack(data_list)
        return data_dict


if __name__ == "__main__":
    images = tf.random.uniform((8, 2, 20, 376, 376))
    labels = tf.random.uniform(
        (8, 1, 20, 376, 376), minval=0, maxval=2, dtype=tf.int32
    )
    data_dict = {"data": images, "seg": labels}
    print(
        data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
    )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)
    data_dict = Convert3DTo2DTransform()(**data_dict)
    print(
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
    print(
        data_dict["data"].shape, data_dict["seg"].shape
    )  # (8, 40, 376, 376) (8, 20, 376, 376)
    data_dict = Convert2DTo3DTransform()(**data_dict)
    print(
        data_dict.keys(), data_dict["data"].shape, data_dict["seg"].shape
    )  # (8, 2, 20, 376, 376) (8, 1, 20, 376, 376)

    images = tf.random.uniform((1, 2, 2, 2, 2))
    labels = tf.random.uniform(
        (1, 1, 2, 2, 2), minval=0, maxval=3, dtype=tf.int32
    )
    data_dict = {"data": images, "target": labels}
    print(
        data_dict["data"].shape, data_dict["target"].shape
    )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    print(data_dict["target"])
    data_dict = ConvertSegmentationToRegionsTransform(
        {"0": (1, 2), "1": (2,)}, "target", "target"
    )(**data_dict)
    print(
        data_dict["data"].shape, data_dict["target"].shape
    )  # (1, 2, 2, 2, 2) (1, 2, 2, 2, 2)
    print(data_dict["target"])

    images = tf.random.uniform((1, 2, 2, 2, 2))
    labels = (
        tf.random.uniform((1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.int32)
        - 1
    )
    data_dict = {"data": images, "seg": labels}
    print(
        data_dict["data"].shape, data_dict["seg"].shape
    )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    print(data_dict)
    data_dict = MaskTransform(
        [(0, False), (1, False)], mask_idx_in_seg=0, set_outside_to=0
    )(**data_dict)
    print(
        data_dict["data"].shape, data_dict["seg"].shape
    )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    print(data_dict)
