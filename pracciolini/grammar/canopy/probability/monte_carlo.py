from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def expectation(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the expected value (mean) of bits set to 1 in the input tensor per probability.

    Args:
        x (tf.Tensor): Input tensor containing packed binary data.
                       Shape: [batch_size, n_probs], dtype uint8.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The expected mean, variance, and total number of bits.
                                                expected_value and variance shapes: [n_probs], dtype tf.float64.
                                                num_samples is scalar, dtype tf.float64.
    """
    # Count the number of bits set to 1 in each element
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    # Sum all the counts to get total number of one-bits
    total_one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=tf.uint64), axis=0) #axis=0
    # Get the total number of elements in the tensor (the number of samples per dimension)
    total_elements = tf.cast(x=tf.shape(x)[0], dtype=tf.uint64)
    # Get the size of each element in bytes
    words_per_element = tf.constant(value=x.dtype.size, dtype=tf.uint64)
    # Convert size to bits (1 byte = 8 bits)
    bits_per_element = tf.math.multiply(x=words_per_element, y=tf.constant(value=8, dtype=tf.uint64))
    # Compute total number of bits in the tensor
    total_bits = tf.math.multiply(x=bits_per_element, y=total_elements)
    num_samples = tf.cast(total_bits, dtype=tf.float64)
    # Compute expected mean value
    expected_value = tf.math.divide(x=tf.cast(x=total_one_bits, dtype=tf.float64), y=num_samples)
    # Compute the variance
    variance = variance_from_mean(expected_value, num_samples)
    return expected_value, variance, num_samples

@tf.function
def variance_from_mean(expected_value: tf.Tensor, num_samples: tf.Tensor) -> tf.Tensor:
    one_minus_mean = tf.math.subtract(x=tf.constant(1.0, dtype=expected_value.dtype), y=expected_value)
    variance = tf.math.multiply(x=expected_value, y=one_minus_mean)
    return variance

@tf.function
def standard_error_from_variance(variance: tf.Tensor, num_samples: tf.Tensor) -> tf.Tensor:
    """
    Computes the standard error from variance and number of samples.

    Args:
        variance (tf.Tensor): Variance values. Shape: [n_probs], dtype tf.float64.
        num_samples (tf.Tensor): Total number of samples. Scalar, dtype tf.float64.

    Returns:
        tf.Tensor: Standard error values. Shape: [n_probs], dtype tf.float64.
    """
    squared = tf.math.divide(x=variance, y=num_samples)
    std_err = tf.math.sqrt(squared)
    return std_err

@tf.function
def expectation_with_confidence_interval(x: tf.Tensor, confidence_level: float = 0.95, dtype=tf.float64) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the expected value (mean) with confidence intervals for the input tensor.

    Args:
        x (tf.Tensor): Input tensor containing packed binary data.
                       Shape: [batch_size, n_probs], dtype uint8.
        confidence_level (float): Desired confidence level (e.g., 0.95 for 95% confidence).
        dtype: tf.float64.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The expected mean, lower limit, and upper limit of the confidence interval.
                                                Shapes: [n_probs], dtype tf.float64.
    """
    expected_value, variance, num_samples = expectation(x)
    # Compute standard error
    std_err = standard_error_from_variance(variance, num_samples)
    # Calculate the z-score for the given confidence level
    dist = tfp.distributions.Normal(loc=tf.constant(0, dtype=dtype), scale=tf.constant(1, dtype=dtype))
    one_plus_conf = tf.math.add(tf.constant(1, dtype=dtype), tf.constant(confidence_level, dtype=dtype))
    one_plus_conf_div_by_2 = tf.math.divide(x=one_plus_conf, y=tf.constant(2, dtype=dtype))
    z = dist.quantile(one_plus_conf_div_by_2)
    # Calculate the margin of error
    margin_of_error = tf.math.multiply(x=z, y=std_err)
    # Calculate confidence interval limits
    lower_limit =tf.math.subtract(x=expected_value, y=margin_of_error)
    lower_limit_clipped = tf.clip_by_value(lower_limit, clip_value_min=0, clip_value_max=1)
    upper_limit = tf.math.add(x=expected_value, y=margin_of_error)
    upper_limit_clipped = tf.clip_by_value(upper_limit, clip_value_min=0, clip_value_max=1)
    # Ensure limits are within [0, 1]
    return lower_limit_clipped, expected_value, upper_limit_clipped


@tf.function
def variational_loss(x: tf.Tensor):
    raise NotImplementedError("variational_loss implementation pending")
