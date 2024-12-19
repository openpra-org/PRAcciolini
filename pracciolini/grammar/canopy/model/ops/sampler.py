from typing import List, Tuple

import tensorflow as tf

@tf.function(jit_compile=True)
def _compute_bits_in_dtype(tensor_type: tf.DType):
    return tf.dtypes.as_dtype(tensor_type).size * 8

@tf.function(jit_compile=True)
def _compute_sample_shape(probs: tf.Tensor,       # [batch_size, num_events].
                          n_sample_packs_per_probability: tf.int32,
                          bitpack_dtype: tf.DType,
                          ) -> Tuple[List, List]:
    """
    Generates bit-packed Bernoulli random variables based on input probabilities.
        Args:
        probs (tf.Tensor): Tensor of probabilities with shape [batch_size, num_events].
        n_sample_packs_per_probability (int): Number of sample packs to generate per probability.
        bitpack_dtype (tf.DType): Data type for bit-packing (e.g., tf.uint8).
    """
    batch_size = tf.cast(tf.shape(probs)[0], dtype=tf.int32)
    num_events = tf.cast(tf.shape(probs)[1], dtype=tf.int32)
    num_bits_per_pack = tf.cast(_compute_bits_in_dtype(bitpack_dtype), dtype=tf.int32)
    num_bits = tf.math.multiply(x=num_bits_per_pack, y=n_sample_packs_per_probability)
    # shape for sampling
    sample_shape = tf.cast([batch_size, num_events, num_bits], dtype=tf.int32)
    # Reshape samples to prepare for bit-packing
    samples_reshaped = [batch_size, num_events, n_sample_packs_per_probability, num_bits_per_pack]
    return sample_shape, samples_reshaped

@tf.function(jit_compile=True)
def _compute_bit_positions(bitpack_dtype: tf.DType):
    num_bits = _compute_bits_in_dtype(bitpack_dtype)
    positions = tf.range(num_bits, dtype=tf.int32)
    positions = tf.cast(positions, bitpack_dtype)
    positions = tf.reshape(positions, [1, 1, -1])  # Shape: [1, 1, num_bits]
    return positions

@tf.function(jit_compile=True)
def generate_bernoulli(
    probs: tf.Tensor,
    n_sample_packs_per_probability: tf.int32,
    bitpack_dtype: tf.DType,
    dtype: tf.DType = tf.float64,
    seed: int = None,
) -> tf.Tensor:
    """
    Generates bit-packed Bernoulli random variables based on input probabilities.

    Args:
        probs (tf.Tensor): Tensor of probabilities with shape [batch_size, num_events].
        n_sample_packs_per_probability (int): Number of sample packs to generate per probability.
        bitpack_dtype (tf.DType): Data type for bit-packing (e.g., tf.uint8).
        dtype (tf.DType, optional): Data type for sampling. Defaults to tf.float64.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        tf.Tensor: Bit-packed tensor of Bernoulli samples with shape [batch_size, num_events].
    """
    sample_shape, samples_bitpack_reshape = _compute_sample_shape(probs=probs,
                                         n_sample_packs_per_probability=n_sample_packs_per_probability,
                                         bitpack_dtype=bitpack_dtype)

    # sample_shape = [batch_size, num_events, n_sample_packs_per_probability * bitpack_dtype * 8].
    # Prepare probabilities to match the shape of 'dist'
    probs_cast = tf.cast(probs, dtype=dtype)
    probs_expanded = tf.expand_dims(probs_cast, axis=-1)  # Shape: [batch_size, num_events, 1]

    # Generate uniform random values
    dist = tf.random.uniform(shape=sample_shape, minval=0, maxval=1, dtype=dtype, seed=seed)
    # Generate Bernoulli samples
    samples = tf.cast(tf.math.less(x=dist, y=probs_expanded), dtype=bitpack_dtype)  # Shape: [batch_size, num_events, num_bits]
    # Reshape samples to prepare for bit-packing
    samples_reshaped = tf.reshape(samples, samples_bitpack_reshape) # Shape: [batch_size, num_events, n_sample_packs_per_probability, num_bits_per_pack]

    # Compute bit positions using the helper function
    positions = _compute_bit_positions(bitpack_dtype)  # Shape: [1, 1, 1, num_bits_per_pack]
    # Shift bits accordingly
    shifted_bits = tf.bitwise.left_shift(samples_reshaped, positions)  # Same shape as samples_reshaped
    # Sum over bits to get packed integers
    packed_bits = tf.reduce_sum(shifted_bits, axis=-1)  # Shape: [batch_size, num_events, n_sample_packs_per_probability]

    # Return the packed bits
    return packed_bits  # Output tensor with shape [batch_size, num_events, n_sample_packs_per_probability]

@tf.function(jit_compile=True)
def generate_bernoulli_no_bitpack(
    probs: tf.Tensor,
    n_samples: tf.int32,
    dtype: tf.DType = tf.float32,
    seed: int = None,
) -> tf.Tensor:
    """
    Generates Bernoulli random variables based on input probabilities.

    Args:
        probs (tf.Tensor): Tensor of probabilities with shape [batch_size, num_events].
        n_samples (int): Number of samples to generate per probability.
        dtype (tf.DType, optional): Data type for sampling. Defaults to tf.float32.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        tf.Tensor: Tensor of Bernoulli samples with shape [batch_size, num_events, n_samples].
    """

    batch_size = tf.shape(probs)[0]
    num_events = tf.shape(probs)[1]
    sample_shape = [batch_size, num_events, n_samples]

    # Prepare probabilities for broadcasting
    probs_cast = tf.cast(probs, dtype=dtype)
    probs_expanded = tf.expand_dims(probs_cast, axis=-1)  # Shape: [batch_size, num_events, 1]

    # Generate uniform random values
    dist = tf.random.uniform(shape=sample_shape, minval=0, maxval=1, dtype=dtype, seed=seed)

    # Generate Bernoulli samples using broadcasting
    samples = tf.cast(dist < probs_expanded, dtype=tf.uint8)  # Shape: [batch_size, num_events, n_samples]

    return samples  # Each sample is 0 or 1, stored as int8