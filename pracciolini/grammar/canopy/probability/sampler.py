from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from pracciolini.grammar.canopy.probability.bitpack import pack_tensor_bits


@tf.function
def generate_uniform_samples(
    dim: Tuple | tf.TensorShape,
    low: float = 0,
    high: float = 1,
    seed: int = 372,
    dtype: tf.DType = tf.float64
) -> tf.Tensor:
    """
    Generates uniformly distributed samples within a specified range and shape.

    This function creates a uniform distribution using TensorFlow Probability and samples
    values based on the provided dimensions, seed, and data type.

    Args:
        dim (Tuple | tf.TensorShape): The shape of the desired sample tensor.
        low (float, optional): The lower bound of the uniform distribution. Defaults to 0.
        high (float, optional): The upper bound of the uniform distribution. Defaults to 1.
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 372.
        dtype (tf.DType, optional): The data type of the generated samples. Defaults to tf.float64.

    Returns:
        tf.Tensor: A tensor containing uniformly distributed samples.
    """
    uniform_dist = tfp.distributions.Uniform(low=tf.cast(low, dtype=dtype), high=tf.cast(high, dtype=dtype))
    uniform_samples = uniform_dist.sample(sample_shape=dim, seed=seed)
    return uniform_samples

@tf.function
def bit_pack_samples(prob, num_samples = int(2 ** 10), seed=372, sampler_dtype=tf.float32, bitpack_dtype=tf.uint8) -> tf.Tensor:
    bits_per_packed_dtype = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    sample_count = int(((num_samples + bits_per_packed_dtype - 1) // bits_per_packed_dtype) * bits_per_packed_dtype)
    samples_dim = (1, sample_count)
    samples = generate_uniform_samples(samples_dim, seed=seed, dtype=sampler_dtype)
    probability = tf.constant(value=prob, dtype=sampler_dtype)
    sampled_probabilities: tf.Tensor = probability >= samples
    packed_samples = pack_tensor_bits(sampled_probabilities, dtype=bitpack_dtype)
    return packed_samples
