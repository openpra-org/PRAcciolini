from typing import List

import tensorflow as tf

@tf.function
def _compute_bits_in_dtype(tensor_type: tf.DType):
    return tf.dtypes.as_dtype(tensor_type).size * 8

@tf.function
def _compute_sample_shape(probs: tf.Tensor,
                          count: tf.int32,
                          bitpack_dtype: tf.DType,
                          ) -> List:
    batch_size = tf.cast(count, dtype=tf.int32)
    num_events = tf.shape(probs)[0] # tf.shape(probs)[0]
    num_bits = _compute_bits_in_dtype(bitpack_dtype)
    return tf.cast([batch_size, num_events, num_bits], dtype=tf.int32)

@tf.function
def _compute_bit_positions(bitpack_dtype: tf.DType):
    num_bits = _compute_bits_in_dtype(bitpack_dtype)
    positions = tf.range(num_bits, dtype=tf.int32)
    positions = tf.cast(positions, bitpack_dtype)
    positions = tf.reshape(positions, [1, 1, -1])  # Shape: [1, 1, num_bits]
    return positions

@tf.function
def generate_bernoulli(probs: tf.Tensor,
                       count: tf.int32, # [batch_size, num_events, num_bits]
                       bitpack_dtype: tf.DType,
                       dtype: tf.DType = tf.float64,
                       seed: int | tf.DType = None,
                       ) -> tf.Tensor:
    shape = _compute_sample_shape(probs, count, bitpack_dtype)
    dist = tf.random.uniform(shape=shape, minval=0, maxval=1, dtype=dtype, seed=seed)
    prob_reshaped = tf.cast(tf.reshape(probs, [1, shape[1], 1]), dtype=dtype)
    # Perform comparison to generate samples
    samples = tf.cast(dist < prob_reshaped, dtype=bitpack_dtype)
    # Shift bits accordingly
    shifted_bit_positions = _compute_bit_positions(bitpack_dtype)
    shifted_bits = tf.bitwise.left_shift(samples, shifted_bit_positions)
    # Sum over bits to get packed integers
    packed_bits = tf.reduce_sum(shifted_bits, axis=-1)  # Shape: [count, len(probs)]
    return packed_bits  # Output tensor of shape [count, len(probs)], dtype bitpack_dtype