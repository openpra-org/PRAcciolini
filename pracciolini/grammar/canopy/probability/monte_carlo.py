from typing import Tuple

import tensorflow as tf

from pracciolini.grammar.canopy.utils import tensor_as_formatted_bit_vectors


@tf.function
def expectation(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes the expected value (mean) of bits set to 1 in the input tensor per probability.

    Args:
        x (tf.Tensor): Input tensor containing packed binary data.
                       Shape: [batch_size, n_probs], dtype uint8.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: The expected mean and total number of bits.
                                     expected_value shape: [n_probs], dtype tf.float64.
                                     num_samples is scalar, dtype tf.float64.
    """
    #sample_shape = [batch_size, num_events, n_sample_packs_per_probability].
    # Count the number of bits set to 1 in each element
    all_bits = tf.cast(tf.shape(x)[-1], dtype=tf.float64) * tf.constant(value=8, dtype=tf.float64) * tf.cast(x.dtype.size, dtype=tf.float64)
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=tf.float64), axis=-1)
    expected_value = one_bits / all_bits
    return expected_value, all_bits

@tf.function
def count_bits(x: tf.Tensor, dtype=tf.uint32) -> Tuple[tf.Tensor, tf.Tensor]:
    #sample_shape = [batch_size, num_events, n_sample_packs_per_probability].
    # Count the number of bits set to 1 in each element
    all_bits = tf.cast(tf.shape(x)[-1], dtype=dtype) * tf.constant(value=8, dtype=dtype) * tf.cast(x.dtype.size, dtype=dtype)
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=dtype), axis=-1)
    return one_bits, all_bits

@tf.function
def count_one_bits(x: tf.Tensor, dtype=tf.uint32) -> Tuple[tf.Tensor, tf.Tensor]:
    #sample_shape = [batch_size, num_events, n_sample_packs_per_probability].
    # Count the number of bits set to 1 in each element
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=dtype), axis=-1)
    return one_bits

@tf.function
def variance_from_mean(expected_value: tf.Tensor) -> tf.Tensor:
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
def expectation_with_confidence_interval(x: tf.Tensor, dtype=tf.float64) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the expected value (mean) with confidence intervals for the input tensor.

    Args:
        x (tf.Tensor): Input tensor containing packed binary data.
                       Shape: [batch_size, n_probs], dtype uint8.
        dtype: tf.float64.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: The lower limit, expected mean, and upper limit of the confidence interval.
                                                Shapes: [n_probs], dtype tf.float64.
                                                num_samples is scalar, dtype tf.float64.
    """
    expected_value, num_samples = expectation(x=x)
    # Compute standard error directly
    variance = expected_value * (1 - expected_value)
    std_err = tf.math.sqrt(variance / num_samples)
    # Use precomputed Z-score if available, else calculate
    z_score_p_95 = tf.constant(value=1.959963984540054, dtype=dtype)
    # Calculate the margin of error
    margin_of_error = z_score_p_95 * std_err
    # Calculate confidence interval limits and clip to [0, 1]
    lower_limit = tf.clip_by_value(expected_value - margin_of_error, 0.0, 1.0)
    upper_limit = tf.clip_by_value(expected_value + margin_of_error, 0.0, 1.0)
    return lower_limit, expected_value, upper_limit

@tf.function
def variational_loss(x: tf.Tensor):
    raise NotImplementedError("variational_loss implementation pending")

@tf.function
def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor, dtype: tf.DType = tf.float64) -> tf.Tensor:
    """
    Custom Mean Squared Error loss function that operates in float64 precision.

    Args:
        y_true (tf.Tensor): Ground truth values. Shape: [batch_size, n_events], dtype can be any type.
        y_pred (tf.Tensor): Predicted values. Shape: [batch_size, n_events], dtype can be any type.
        dtype (tf.DType): dtype can be any float type, defaults to tf.float64
    Returns:
        tf.Tensor: Scalar loss value. defaults to tf.float64
    """
    y_true = tf.cast(y_true, dtype=dtype)
    y_pred = tf.cast(y_pred, dtype=dtype)
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return loss