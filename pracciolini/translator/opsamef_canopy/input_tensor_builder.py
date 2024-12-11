from collections import OrderedDict
from typing import Tuple, Any

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Operation
from tensorflow.python.framework.ops import _EagerTensorBase

from pracciolini.translator.opsamef_canopy.sampler import pack_tensor_bits


def build_probabilities_from_event_map(
    event_map: OrderedDict[str, str],
    name: str = "X",
    dtype: tf.DType = tf.float64
) -> Tuple[list, Operation | _EagerTensorBase]:
    """
    Generates a list of event names and a TensorFlow constant tensor representing their probabilities.

    This function converts the event values from the event map to floating-point numbers and
    creates a TensorFlow constant tensor with the specified name and data type.

    Args:
        event_map (Dict[str, str]): A mapping of event names to their string values.
        name (str, optional): The name assigned to the TensorFlow constant tensor. Defaults to "P(x)".
        dtype (tf.DType, optional): The data type of the TensorFlow constant. Defaults to tf.float64.

    Returns:
        Tuple[list, Operation | _EagerTensorBase]:
            - A list of event names.
            - A TensorFlow constant tensor containing the probability values.
    """
    cast_values = [float(value) for value in event_map.values()]
    probabilities = tf.constant(value=cast_values, name=name, dtype=dtype)
    return list(event_map.keys()), probabilities


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


def sample_probabilities_bit_packed(
    probabilities: tf.Tensor,
    num_samples: int = int(2 ** 20),
    seed: int = 372,
    sampler_dtype: tf.DType = tf.float32,
    bitpack_dtype: tf.DType = tf.uint8
) -> Tuple[tf.Tensor, OrderedDict[str, Any]]:
    """
    Samples probabilities and packs the sampled bits into a compact format.

    This function generates uniform samples, compares them against the provided probabilities
    to determine sampled events, and then packs the resulting boolean tensor into bit-packed format.

    Args:
        probabilities (tf.Tensor): A tensor of probabilities for each event.
        num_samples (int, optional): The number of samples to generate. Defaults to int(1e6).
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 372.
        sampler_dtype (tf.DType, optional): Data type for the sampler tensor. Defaults to tf.float32.
        bitpack_dtype (tf.DType, optional): Data type for the bit-packed samples. Defaults to tf.uint8.

    Returns:
        Tuple[Tensor, Dict[str, Any]]:
            - A tensor containing the bit-packed sampled probabilities.
            - A dictionary with metadata about the sampling process, including seed, number of samples,
              data types, and sampler shape.
    """
    bits_per_packed_dtype = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    # Calculate the number of samples rounded up to the nearest multiple of bits_per_packed_dtype
    sample_count = int(((num_samples + bits_per_packed_dtype - 1) // bits_per_packed_dtype) * bits_per_packed_dtype)
    samples_dim = (probabilities.shape[0], sample_count)
    samples = generate_uniform_samples(samples_dim, seed=seed, dtype=sampler_dtype)
    sampled_probabilities: tf.Tensor = probabilities[:, tf.newaxis] >= samples
    packed_samples = pack_tensor_bits(sampled_probabilities, dtype=bitpack_dtype)
    return packed_samples, OrderedDict({
        "seed": seed,
        "num_samples": sample_count,
        "bitpack_dtype": packed_samples.dtype,
        "sampler_dtype": samples.dtype,
        "sampler_shape": samples.shape
    })


def bit_pack_samples(prob, num_samples = int(2 ** 10), seed=372, sampler_dtype=tf.float32, bitpack_dtype=tf.uint8):
    bits_per_packed_dtype = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    sample_count = int(((num_samples + bits_per_packed_dtype - 1) // bits_per_packed_dtype) * bits_per_packed_dtype)
    samples_dim = (1, sample_count)
    samples = generate_uniform_samples(samples_dim, seed=seed, dtype=sampler_dtype)
    probability = tf.constant(value=prob, dtype=sampler_dtype)
    sampled_probabilities: tf.Tensor = probability >= samples
    packed_samples = pack_tensor_bits(sampled_probabilities, dtype=bitpack_dtype)
    return packed_samples
