import tensorflow as tf

from pracciolini.grammar.canopy.probability.distributions import Bernoulli


@tf.function
def expectation(x: tf.Tensor) -> tf.Tensor:
    """
    Computes the expected value (mean) of bits set to 1 in the input tensor.

    Args:
        x (tf.Tensor): Input tensor containing binary data.

    Returns:
        tf.Tensor: The expected mean value.
    """
    # Count the number of bits set to 1 in each element
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    # Sum all the counts to get total number of one-bits
    total_one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=tf.uint64), axis=None) #axis=0
    # Get the total number of elements in the tensor
    total_elements = tf.cast(x=tf.size(input=x, out_type=tf.int64), dtype=tf.uint64)
    # Get the size of each element in bytes
    words_per_element = tf.constant(value=x.dtype.size, dtype=tf.uint64)
    # Convert size to bits (1 byte = 8 bits)
    bits_per_element = tf.math.multiply(x=words_per_element, y=tf.constant(value=8, dtype=tf.uint64))
    # Compute total number of bits in the tensor
    total_bits = tf.math.multiply(x=bits_per_element, y=total_elements)
    # Compute expected mean value
    expected_mean = tf.math.divide(x=tf.cast(x=total_one_bits, dtype=tf.float64), y=tf.cast(x=total_bits, dtype=tf.float64))
    return expected_mean

@tf.function
def variance(x: tf.Tensor) -> tf.Tensor:
    pass

@tf.function
def confidence_intervals(x: tf.Tensor) -> tf.Tensor:
    pass

@tf.function
def variational_loss(x: tf.Tensor) -> tf.Tensor:
    pass


@tf.function
def sample_bernoulli(probs: tf.Tensor, sample_shape, pack_bits_dtype: tf.DType):
    distribution = Bernoulli(probs=probs, dtype=tf.bool, pack_bits_dtype=pack_bits_dtype)
    return distribution.sample(sample_shape=sample_shape)