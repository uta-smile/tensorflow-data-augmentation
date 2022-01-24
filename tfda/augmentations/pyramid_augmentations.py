
import tensorflow as tf

from tfda.data_processing_utils import erosion, dilation, opening, closing, random_choice, update_tf_channel


@tf.function
def augment_apply_random_binary_operation(data, channel_idx, p_per_sample=0.3, strel_size=(1, 10), p_per_label=1):
    return tf.cond(tf.less(tf.random.uniform([]), p_per_sample), 
                lambda: augment_apply_random_binary_operation_wrapper(data, channel_idx, strel_size, p_per_label),
                lambda: data)

@tf.function
def augment_apply_random_binary_operation_wrapper(data, channel_idx, strel_size=(1, 10), p_per_label=1):
    ch = tf.identity(channel_idx)
    ch = tf.random.shuffle(ch)
    c = tf.constant(0)
    cond_to_loop = lambda data, c, ch, strel_size, p_per_label: tf.less(c, tf.shape(ch)[0])
    new_data = tf.while_loop(cond_to_loop, augment_apply_random_binary_operation_per_channel, loop_vars=[data, c, ch, strel_size, p_per_label])
    return new_data

@tf.function
def augment_apply_random_binary_operation_per_channel(data, channel_idx, channel_indexes, strel_size=(1, 10), p_per_label=1):
    return tf.cond(tf.less(tf.random.uniform([]), p_per_label),
                lambda: do_augment_apply_random_binary_operation_per_channel(data, channel_idx, channel_indexes, strel_size),
                lambda: data, channel_idx+1, channel_indexes, strel_size)

@tf.function
def do_augment_apply_random_binary_operation_per_channel(data, channel_idx, channel_indexes, strel_size=(1, 10), any_of_these=(0, 1, 2, 3)):
    operation_idx = random_choice(any_of_these, 0)[0]
    kernel_size = tf.random.uniform([], minval=strel_size[0], maxval=strel_size[1], dtype=tf.int64)
    ch = channel_indexes[channel_idx]
    workon = tf.cast(data[ch], tf.int64)
    res = do_operation(workon, kernel_size, operation_idx)
    new_data = update_tf_channel(data, ch, res)

    other_indexes = tf.gather(channel_indexes, tf.where(tf.logical_not(tf.equal(channel_indexes, ch))), axis=0)
    other_idx = tf.constant(0)
    cond_to_loop = lambda new_data, res, workon, other_idx, other_indexes: tf.less(other_idx, tf.shape(other_indexes)[0])
    new_data = tf.while_loop(cond_to_loop, do_rest_operation, loop_vars=[new_data, res, workon, other_idx, other_indexes])

    return new_data, channel_idx+1, channel_indexes, strel_size

@tf.function
def do_rest_operation(data, res, workon, other_idx, other_indexes):
    oi = other_indexes[other_idx]
    was_added_mask = tf.where(tf.greater(res - workon, 0))
    other_data = tf.tensor_scatter_nd_update(data[oi], was_added_mask, tf.zeros(tf.shape(was_added_mask)[0]))
    new_data = update_tf_channel(data, oi, other_data)

    return new_data, res, workon, other_idx+1, other_indexes

@tf.function
def do_operation(data, kernel_size, branch_idx):
    return tf.switch_case(branch_idx, branch_fns={0: lambda: erosion(data, kernel_size), 1: lambda: dilation(data, kernel_size), 2: lambda: opening(data, kernel_size), 3: lambda: closing(data, kernel_size)})

@tf.function
def augment_move_seg_as_onehot_data_batch(seg, all_seg_labels):
    seg_channel = tf.map_fn(
        lambda c: augment_move_seg_as_onehot_data_channel(seg, all_seg_labels[c]),
        elems=tf.range(tf.shape(all_seg_labels)[0]),
        dtype=tf.float32
    )
    return seg_channel

@tf.function
def augment_move_seg_as_onehot_data_channel(seg, seg_label):
    indices = tf.where(tf.equal(seg, seg_label))
    seg_result = tf.tensor_scatter_nd_update(seg, indices, tf.ones(tf.shape(indices)[0]))
    return seg_result

@tf.function
def do_remove_from_origin(seg, channel_idx):
    return tf.concat([seg[:, :channel_idx], seg[:, channel_idx+1:]], axis=1)
