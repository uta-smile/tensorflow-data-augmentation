from tfda.base import TFDABase
from tfda.defs import DTFT, TFDAData

# Tensorflow
import tensorflow as tf


class RemoveLabelTransform(TFDABase):
    """
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    """

    def __init__(
        self,
        remove_label: tf.Tensor,
        replace_with: tf.Tensor = 0.0,
        input_key: str = "seg",
        output_key: str = "seg",
        **kws,
    ):
        super().__init__(**kws)
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        # seg = data_dict[self.input_key]
        # condition = tf.equal(seg, self.remove_label)
        # case_true = tf.zeros(
        #     tf.shape(data_dict["seg"])
        # )  # self.replace_with = 0
        # case_false = seg
        # seg = tf.where(condition, case_true, case_false)
        # data_dict[self.output_key] = seg
        # return data_dict
        seg = dataset.seg
        return TFDAData(
            dataset.data,
            tf.where(
                tf.equal(seg, self.remove_label), tf.zeros(tf.shape(seg)), seg
            ),
        )


class RenameTransform(TFDABase):
    """Saves the value of data_dict[in_key] to data_dict[out_key].

    Optionally removes data_dict[in_key] from the dict.
    """

    def __init__(self, in_key, out_key, delete_old=False, **kws):
        super().__init__(**kws)
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    @tf.function(experimental_follow_type_hints=True)
    def call(self, data_dict: DTFT) -> DTFT:
        data_dict = data_dict.copy()
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict


if __name__ == "__main__":
    images = tf.random.uniform((1, 2, 2, 2, 2))
    labels = (
        tf.random.uniform(
            (1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.float32
        )
        - 1
    )
    data_dict = TFDAData(images, labels)
    tf.print(
        data_dict["data"].shape, data_dict["seg"].shape
    )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    tf.print(data_dict)
    data_dict = RemoveLabelTransform(-1, 0)(data_dict)
    tf.print(
        data_dict["data"].shape, data_dict["seg"].shape
    )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    tf.print(data_dict)

    # images = tf.random.uniform((1, 2, 2, 2, 2))
    # labels = tf.random.uniform(
    #     (1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.int32
    # )
    # data_dict = {"data": images, "seg": labels}
    # tf.print(
    #     data_dict["data"].shape, data_dict["seg"].shape
    # )  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    # tf.print(data_dict)
    # data_dict = RenameTransform("seg", "target", True)(data_dict)
    # tf.print(data_dict)
