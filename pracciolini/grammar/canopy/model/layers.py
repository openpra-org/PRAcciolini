import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
from functools import reduce

from pracciolini.grammar.canopy.probability.monte_carlo import expectation


class Expectation(Layer):
    """
    Custom Keras Layer that computes the expected value (mean) of bits set to 1 in the input tensor.

    This layer wraps the 'expectation_impl' function to be used within a Keras model.
    """

    def __init__(self, **kwargs):
        super(Expectation, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Invokes the layer on the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after computing the expected value.
        """
        return expectation(inputs)

class BitwiseNot(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNot, self).__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError("BitwiseNot Layer requires exactly one input tensor.")
            inputs = inputs[0]
        return tf.bitwise.invert(inputs)


class BitwiseAnd(Layer):
    def __init__(self, **kwargs):
        super(BitwiseAnd, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseAnd layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseAnd layer requires at least two input tensors.")
        return self.compute_bitwise_and(inputs)

    @tf.function
    def compute_bitwise_and(self, inputs):
        # Stack inputs along a new dimension to create a single tensor.
        stacked_inputs = tf.stack(inputs)
        # Use tf.foldl to perform the bitwise AND reduction efficiently.
        result = tf.foldl(lambda a, b: tf.bitwise.bitwise_and(a, b), stacked_inputs)
        return result

class BitwiseOr(Layer):
    def __init__(self, **kwargs):
        super(BitwiseOr, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseOr layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseOr layer requires at least two input tensors.")
        return self.compute_bitwise_or(inputs)

    @tf.function
    def compute_bitwise_or(self, inputs):
        # Stack inputs along a new dimension to create a single tensor.
        stacked_inputs = tf.stack(inputs)
        # Use tf.foldl to perform the bitwise OR reduction efficiently.
        result = tf.foldl(lambda a, b: tf.bitwise.bitwise_or(a, b), stacked_inputs)
        return result

class BitwiseXor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXor, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseXor Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseXor Layer requires at least two input tensors.")
        return reduce(tf.bitwise.bitwise_xor, inputs)

class BitwiseNand(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNand, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseNand Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseNand Layer requires at least two input tensors.")
        result = reduce(tf.bitwise.bitwise_and, inputs)
        return tf.bitwise.invert(result)

class BitwiseNor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNor, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseNor Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseNor Layer requires at least two input tensors.")
        result = reduce(tf.bitwise.bitwise_or, inputs)
        return tf.bitwise.invert(result)

class BitwiseXnor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXnor, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseXnor Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseXnor Layer requires at least two input tensors.")
        result = reduce(tf.bitwise.bitwise_xor, inputs)
        return tf.bitwise.invert(result)


class BitpackedBernoulli(Layer):
    def __init__(self, probs, batch_size, sampler_dtype=tf.float64, **kwargs):
        super(BitpackedBernoulli, self).__init__(**kwargs)
        self._sampler_dtype = sampler_dtype
        self.probs = tf.constant(value=probs, dtype=self._sampler_dtype)
        self._batch_size = batch_size
        self._bitpack_dtype = self.dtype
        self._bitpack_supported_dtype_limits = [tf.uint8, tf.uint64]
        self._bitpack_bits_per_dtype = tf.dtypes.as_dtype(self._bitpack_dtype).size * 8
        self._bitpack_num_bits_to_sample = self._bitpack_bits_per_dtype
        self._bitpack_bits_per_supported_dtypes = [tf.dtypes.as_dtype(dtype).size * 8 for dtype in self._bitpack_supported_dtype_limits]

        if self._bitpack_bits_per_dtype < self._bitpack_bits_per_supported_dtypes[0]:
            raise ValueError(
                f"Cannot handle word-size {self._bitpack_dtype}, which is smaller than {self._bitpack_bits_per_supported_dtypes[0]}"
            )

        if self._bitpack_bits_per_dtype > self._bitpack_bits_per_supported_dtypes[1]:
            raise ValueError(
                f"Cannot handle word-size {self._bitpack_dtype}, which is larger than {self._bitpack_bits_per_supported_dtypes[1]}"
            )

        if self._batch_size * self._bitpack_num_bits_to_sample > tf.dtypes.int32.max:
            raise ValueError(
                f"Cannot handle batch_size larger than {int(tf.dtypes.int32.max / self._bitpack_num_bits_to_sample)} for dtype {self._bitpack_dtype}"
            )

        # Initialize the Bernoulli distribution
        self._distribution = tfp.distributions.Bernoulli(probs=self.probs, dtype=self._bitpack_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "probs": self.probs.numpy(),
            "batch_size": self._batch_size,
            "sampler_dtype": self._bitpack_dtype,
        })
        return config

    @tf.function
    def call(self, inputs):

        batch_size = tf.cast(self._batch_size, dtype=tf.uint64)

        # Sample bits from the Bernoulli distribution
        samples = self._distribution.sample(
            sample_shape=[batch_size, self._bitpack_num_bits_to_sample]
        )  # Shape: [batch_size, num_bits_to_sample]

        # Create bit positions: [0, 1, 2, ..., num_bits_to_sample - 1]
        bit_positions = tf.range(self._bitpack_num_bits_to_sample, dtype=tf.int32)
        # Cast bit_positions to the same dtype as bit_values
        bit_positions = tf.cast(bit_positions, self._bitpack_dtype)
        bit_positions = tf.reshape(bit_positions, [1, -1])  # Shape: [1, num_bits_to_sample]

        # Convert boolean samples to integer type
        #bit_values = tf.cast(samples, self.bitpack_dtype)  # Shape: [batch_size, num_bits_to_sample]

        # Shift bits accordingly using tf.bitwise.left_shift
        shifted_bits = tf.bitwise.left_shift(samples, bit_positions)  # Corrected line

        # Sum over the bits to get packed integers
        packed_bits = tf.reduce_sum(shifted_bits, axis=-1)  # Shape: [batch_size]

        return packed_bits  # Output tensor of shape [batch_size], dtype self.bitpack_dtype
