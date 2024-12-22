from typing import Tuple, List

import tensorflow as tf


class Sampler(tf.Module):

    def __init__(self, name=None):
        super().__init__(name)

    @staticmethod
    @tf.function(jit_compile=True)
    def _mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor, dtype: tf.DType = tf.float64) -> tf.Tensor:
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

    @staticmethod
    @tf.function(jit_compile=True)
    def _compute_bits_in_dtype(tensor_type: tf.DType):
        return tf.dtypes.as_dtype(tensor_type).size * 8

    @staticmethod
    @tf.function(jit_compile=True)
    def _compute_sample_shape(probs: tf.Tensor,  # [batch_size, num_events].
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
        num_events = tf.cast(tf.shape(probs)[0], dtype=tf.int32)
        batch_size = tf.cast(tf.shape(probs)[1], dtype=tf.int32)
        num_bits_per_pack = tf.cast(Sampler._compute_bits_in_dtype(bitpack_dtype), dtype=tf.int32)
        num_bits = tf.math.multiply(x=num_bits_per_pack, y=n_sample_packs_per_probability)
        # shape for sampling
        sample_shape = tf.cast([num_events, batch_size, num_bits], dtype=tf.int32)
        # Reshape samples to prepare for bit-packing
        samples_reshaped = [num_events, batch_size, n_sample_packs_per_probability, num_bits_per_pack]
        return sample_shape, samples_reshaped

    @staticmethod
    @tf.function(jit_compile=True)
    def _compute_bit_positions(bitpack_dtype: tf.DType):
        num_bits = Sampler._compute_bits_in_dtype(bitpack_dtype)
        positions = tf.range(num_bits, dtype=tf.int32)
        positions = tf.cast(positions, bitpack_dtype)
        positions = tf.reshape(positions, [1, 1, -1])  # Shape: [1, 1, num_bits]
        return positions

    @staticmethod
    @tf.function(jit_compile=True)
    def _generate_bernoulli(
            probs: tf.Tensor,
            n_sample_packs_per_probability: tf.int32,
            bitpack_dtype: tf.DType,
            dtype: tf.DType = tf.float64,
            seed: int = None,
    ) -> tf.Tensor:
        """
        Generates bit-packed Bernoulli random variables based on input probabilities.

        Args:
            probs (tf.Tensor): Tensor of probabilities with shape [num_events, batch_size].
            n_sample_packs_per_probability (int): Number of sample packs to generate per probability.
            bitpack_dtype (tf.DType): Data type for bit-packing (e.g., tf.uint8).
            dtype (tf.DType, optional): Data type for sampling. Defaults to tf.float64.
            seed (int, optional): Random seed. Defaults to None.

        Returns:
            tf.Tensor: Bit-packed tensor of Bernoulli samples with shape [num_events, batch_size].
        """
        sample_shape, samples_bitpack_reshape = Sampler._compute_sample_shape(probs=probs,
                                                                      n_sample_packs_per_probability=n_sample_packs_per_probability,
                                                                      bitpack_dtype=bitpack_dtype)

        # sample_shape = [num_events, batch_size, n_sample_packs_per_probability * bitpack_dtype * 8].
        # Prepare probabilities to match the shape of 'dist'
        probs_cast = tf.cast(probs, dtype=dtype)
        probs_expanded = tf.expand_dims(probs_cast, axis=-1)  # Shape: [num_events, batch_size, 1]

        # Generate uniform random values
        dist = tf.random.uniform(shape=sample_shape, minval=0, maxval=1, dtype=dtype)
        # Generate Bernoulli samples
        samples = tf.cast(tf.math.less(x=dist, y=probs_expanded), dtype=bitpack_dtype)  # Shape: [num_events, batch_size, num_bits]
        # Reshape samples to prepare for bit-packing
        samples_reshaped = tf.reshape(samples, samples_bitpack_reshape)  # Shape: [num_events, batch_size, n_sample_packs_per_probability, num_bits_per_pack]

        # Compute bit positions using the helper function
        positions = Sampler._compute_bit_positions(bitpack_dtype)  # Shape: [1, 1, 1, num_bits_per_pack]
        # Shift bits accordingly
        shifted_bits = tf.bitwise.left_shift(samples_reshaped, positions)  # Same shape as samples_reshaped
        # Sum over bits to get packed integers
        packed_bits = tf.reduce_sum(shifted_bits, axis=-1)  # Shape: [num_events, batch_size, n_sample_packs_per_probability]

        # Return the packed bits
        return packed_bits  # Output tensor with shape [num_events, batch_size, n_sample_packs_per_probability]

    @staticmethod
    @tf.function(jit_compile=True)
    def _count_one_bits(x: tf.Tensor, axis=None, dtype=tf.uint32) -> tf.Tensor:
        # sample_shape = [batch_size, num_events, n_sample_packs_per_probability].
        pop_counts = tf.raw_ops.PopulationCount(x=x)
        one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=dtype), axis=axis)
        return one_bits

    @staticmethod
    @tf.function(jit_compile=True)
    def _p95_ci(means: tf.Tensor, total: tf.Tensor, dtype=tf.float64) -> Tuple[tf.Tensor, tf.Tensor]:
        variance = means * (1 - means)
        std_err = tf.sqrt(variance / total)
        # Use precomputed Z-score if available, else calculate
        z_score_p_95 = tf.constant(value=1.959963984540054, dtype=dtype)
        # Calculate the margin of error
        margin_of_error = z_score_p_95 * std_err
        # Calculate confidence interval limits and clip to [0, 1]
        lower_limit = tf.clip_by_value(means - margin_of_error, 0.0, 1.0)
        upper_limit = tf.clip_by_value(means + margin_of_error, 0.0, 1.0)
        return lower_limit, upper_limit