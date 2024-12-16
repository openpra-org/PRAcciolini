import tensorflow as tf
from tensorflow.keras.layers import Layer

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


@tf.function
def bitwise_nary_op(bin_fn, inputs):
    """
    Applies the bitwise op across the 'num_events' dimension.

    Args:
        bin_fn (function): The bitwise function to be applied.
        inputs (tf.Tensor): Input tensor with shape (batch_size, num_events).

    Returns:
        tf.Tensor: Output tensor with shape (batch_size, 1) after reducing.
    """
    # Transpose the inputs to have shape (num_events, batch_size)
    transposed_inputs = tf.transpose(inputs, perm=[1, 0])

    # Initialize the accumulation with zeros
    initial_value = tf.zeros_like(transposed_inputs[0])

    # Use tf.scan to apply bitwise OP across the num_events dimension
    result = tf.scan(
        fn=lambda a, b: bin_fn(a, b),
        elems=transposed_inputs,
        initializer=initial_value,
    )
    # result has shape (num_events, batch_size)

    # Get the final accumulated result
    final_result = result[-1]  # Shape: (batch_size,)

    # Reshape result to (batch_size, 1)
    final_result = tf.expand_dims(final_result, axis=1)
    return final_result


class BitwiseNot(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNot, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        return tf.bitwise.invert(inputs)


class BitwiseAnd(Layer):
    def __init__(self, **kwargs):
        super(BitwiseAnd, self).__init__(**kwargs)

    def call(self, inputs):
        return bitwise_nary_op(tf.bitwise.bitwise_and, inputs)


class BitwiseOr(Layer):
    def __init__(self, **kwargs):
        super(BitwiseOr, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        return bitwise_nary_op(tf.bitwise.bitwise_or, inputs)


class BitwiseXor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXor, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        return bitwise_nary_op(tf.bitwise.bitwise_xor, inputs)


class BitwiseNand(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNand, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        return tf.bitwise.invert(bitwise_nary_op(tf.bitwise.bitwise_and, inputs))


class BitwiseNor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNor, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        return tf.bitwise.invert(bitwise_nary_op(tf.bitwise.bitwise_or, inputs))


class BitwiseXnor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXnor, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        return tf.bitwise.invert(bitwise_nary_op(tf.bitwise.bitwise_xor, inputs))


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

    def get_config(self):
        config = super().get_config()
        config.update({
            "probs": self.probs.numpy(),
            "batch_size": self._batch_size,
            "sampler_dtype": self._bitpack_dtype,
        })
        return config

    @tf.function
    def compute_bit_positions(self):
        # Create bit positions: [0, 1, ..., num_bits_to_sample -1], shape [1, 1, num_bits_to_sample]
        positions = tf.range(self._bitpack_num_bits_to_sample, dtype=tf.int32)
        positions = tf.cast(positions, self._bitpack_dtype)
        positions = tf.reshape(positions, [1, 1, -1])  # Shape: [1, 1, num_bits_to_sample]
        return positions

    @tf.function
    def call(self, inputs):
        batch_size = tf.cast(self._batch_size, dtype=tf.int32)
        num_events = tf.shape(self.probs)[0]
        num_bits = tf.cast(self._bitpack_num_bits_to_sample, dtype=tf.int32)
        dist = tf.random.uniform(shape=[batch_size, num_events, num_bits], minval=0, maxval=1, dtype=self._sampler_dtype)
        probs = tf.reshape(self.probs, [1, num_events, 1])
        # Perform comparison to generate samples
        samples = tf.cast(dist < probs, dtype=self._bitpack_dtype)
        # Shift bits accordingly
        shifted_bits = tf.bitwise.left_shift(samples, self.compute_bit_positions())
        # Sum over bits to get packed integers
        packed_bits = tf.reduce_sum(shifted_bits, axis=-1)  # Shape: [batch_size, len(probs)]
        return packed_bits  # Output tensor of shape [batch_size, len(probs)], dtype self.bitpack_dtype
