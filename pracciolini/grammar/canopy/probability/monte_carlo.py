from typing import Tuple

import tensorflow as tf

from pracciolini.grammar.canopy.model.ops.sampler import generate_bernoulli


@tf.function(jit_compile=True)
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

#@tf.function(jit_compile=True)
def count_bits(x: tf.Tensor, axis=None, dtype=tf.uint32) -> Tuple[tf.Tensor, tf.Tensor]:
    #sample_shape = [batch_size, num_events, n_sample_packs_per_probability].
    # Count the number of bits set to 1 in each element
    all_bits = tf.cast(tf.shape(x)[-1], dtype=dtype) * tf.constant(value=8, dtype=dtype) * tf.cast(x.dtype.size, dtype=dtype)
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=dtype), axis=axis)
    return one_bits, all_bits

@tf.function(jit_compile=True)
def count_one_bits(x: tf.Tensor, axis=None, dtype=tf.uint32) -> tf.Tensor:
    #sample_shape = [batch_size, num_events, n_sample_packs_per_probability].
    # Count the number of bits set to 1 in each element
    pop_counts = tf.raw_ops.PopulationCount(x=x)
    one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=dtype), axis=axis)
    return one_bits

@tf.function(jit_compile=True)
def variance_from_mean(expected_value: tf.Tensor) -> tf.Tensor:
    one_minus_mean = tf.math.subtract(x=tf.constant(1.0, dtype=expected_value.dtype), y=expected_value)
    variance = tf.math.multiply(x=expected_value, y=one_minus_mean)
    return variance

@tf.function(jit_compile=True)
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

@tf.function(jit_compile=True)
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

@tf.function(jit_compile=True)
def variational_loss(x: tf.Tensor):
    raise NotImplementedError("variational_loss implementation pending")

@tf.function(jit_compile=True)
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


@tf.function(jit_compile=True)
def batched_estimate(probs_: tf.Tensor,
                     num_batches_: int,
                     sample_size_: int,
                     bitpack_dtype_: tf.DType,
                     sampler_dtype_: tf.DType,
                     acc_dtype_: tf.DType = tf.uint64):
    # Get dynamic shapes
    batch_size_ = tf.shape(probs_)[0]
    num_events_ = tf.shape(probs_)[1]

    # Convert sample_size_ and num_batches_ to tensors if needed
    num_batches_tensor = tf.constant(num_batches_, dtype=tf.int32)
    sample_size_tensor = tf.constant(sample_size_, dtype=acc_dtype_)

    # Get bitpack size in bytes as tensor
    bitpack_size_bytes = tf.constant(tf.dtypes.as_dtype(bitpack_dtype_).size, dtype=acc_dtype_)

    # Compute event dimension size
    event_dim_size_ = sample_size_tensor * bitpack_size_bytes * tf.constant(8, dtype=acc_dtype_)
    event_bits_in_batch_ = event_dim_size_

    # Initialize tensors
    cumulative_one_bits_ = tf.zeros([batch_size_, num_events_], dtype=acc_dtype_)
    losses_ = tf.TensorArray(dtype=tf.float64, size=num_batches_tensor)

    for batch_idx_ in tf.range(num_batches_tensor):
        # Generate samples
        packed_bits_ = generate_bernoulli(
            probs=probs_,
            n_sample_packs_per_probability=sample_size_,
            bitpack_dtype=bitpack_dtype_,
            dtype=sampler_dtype_,
        )

        # logic/tree here

        # Compute the number of one-bits
        one_bits_in_batch_ = tf.reduce_sum(
            tf.cast(
                tf.raw_ops.PopulationCount(x=packed_bits_),
                dtype=acc_dtype_
            ),
            axis=-1
        )
        # Update cumulative counts
        cumulative_one_bits_ += one_bits_in_batch_
        cumulative_bits_ = (tf.cast(batch_idx_ + 1, dtype=acc_dtype_)) * event_bits_in_batch_
        # Compute expected values
        updated_batch_expected_value_ = tf.cast(cumulative_one_bits_, dtype=tf.float64) / tf.cast(
            cumulative_bits_, dtype=tf.float64)
        updated_expected_value_ = tf.reduce_mean(updated_batch_expected_value_, axis=0)
        # Compute loss
        updated_batch_loss_ = mse_loss(probs_, updated_expected_value_)
        # Write to TensorArray
        losses_ = losses_.write(batch_idx_, updated_batch_loss_)

    return losses_.stack(), updated_expected_value_