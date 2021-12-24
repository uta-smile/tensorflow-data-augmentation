import tensorflow as tf


class RemoveLabelTransform(AbstractTransform):
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    @tf.function
    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        condition = tf.equal(seg, self.remove_label)
        case_true = tf.zeros_like(data_dict['seg'])  # self.replace_with = 0
        case_false = seg
        seg = tf.where(condition, case_true, case_false)
        data_dict[self.output_key] = seg
        return data_dict
      
      
class RenameTransform(AbstractTransform):
    '''
    Saves the value of data_dict[in_key] to data_dict[out_key]. Optionally removes data_dict[in_key] from the dict.
    '''

    def __init__(self, in_key, out_key, delete_old=False):
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    @tf.function
    def __call__(self, **data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict
      
      
if __name__ == "__main__":
    images = tf.random.uniform((1, 2, 2, 2, 2))
    labels = tf.random.uniform((1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.int32) - 1
    data_dict = {'data': images, 'seg': labels}
    print(data_dict['data'].shape, data_dict['seg'].shape)  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    print(data_dict)
    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)
    print(data_dict['data'].shape, data_dict['seg'].shape)  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    print(data_dict)
    
    
    images = tf.random.uniform((1, 2, 2, 2, 2))
    labels = tf.random.uniform((1, 1, 2, 2, 2), minval=0, maxval=2, dtype=tf.int32)
    data_dict = {'data': images, 'seg': labels}
    print(data_dict['data'].shape, data_dict['seg'].shape)  # (1, 2, 2, 2, 2) (1, 1, 2, 2, 2)
    print(data_dict)
    data_dict = RenameTransform('seg', 'target', True)(**data_dict)
    print(data_dict)
      
