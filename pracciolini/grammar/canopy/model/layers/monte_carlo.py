import tensorflow as tf

from pracciolini.grammar.canopy.probability.monte_carlo import expectation


class BernoulliSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, prob, output_shape, dtype=tf.uint8, **kwargs):
        super(BernoulliSamplingLayer, self).__init__(**kwargs)
        self.prob = prob
        self.output_shape = output_shape
        self.dtype = dtype

    def call(self, inputs=None):
        # Generate random values
        random_vals = tf.random.uniform(shape=self.output_shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        # Generate samples by comparing random values with the probability
        samples = tf.cast(random_vals < self.prob, dtype=self.dtype)
        return samples


class BitpackedBernoulliSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, prob, sample_shape, pack_bits_dtype=tf.uint8, **kwargs):
        super(BitpackedBernoulliSamplingLayer, self).__init__(**kwargs)
        self.prob = prob
        self.sample_shape = sample_shape
        self.pack_bits_dtype = pack_bits_dtype

        # Determine bitpacking properties
        self.bitpack_largest_supported_dtype = tf.uint64
        self.bitpack_bits_per_dtype = tf.dtypes.as_dtype(self.pack_bits_dtype).size * 8
        self.bitpack_bits_per_largest_supported_dtype = tf.dtypes.as_dtype(self.bitpack_largest_supported_dtype).size * 8

        if self.bitpack_bits_per_dtype > self.bitpack_bits_per_largest_supported_dtype:
            raise ValueError(
                f"Cannot handle word-size {self.pack_bits_dtype}, which is larger than {self.bitpack_largest_supported_dtype}"
            )

        # Precompute weights for bitpacking
        weights = [1 << (self.bitpack_bits_per_dtype - 1 - i) for i in range(self.bitpack_bits_per_dtype)]
        compute_dtype = self.bitpack_largest_supported_dtype
        self._weights = tf.constant(weights, dtype=compute_dtype)

    def call(self, inputs=None):
        # Generate samples
        compute_dtype = self.bitpack_largest_supported_dtype
        bitpack_bits_per_dtype = self.bitpack_bits_per_dtype

        # Determine the extended shape including the bits to pack
        samples_shape = tf.concat([self.sample_shape, [bitpack_bits_per_dtype]], axis=0)
        random_vals = tf.random.uniform(shape=samples_shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        bool_tensor = tf.less(random_vals, self.prob)
        bool_tensor = tf.reshape(bool_tensor, [-1, bitpack_bits_per_dtype])  # [batch_size, bits_per_dtype]

        # Bitpack the boolean tensor
        bits = tf.cast(bool_tensor, dtype=compute_dtype)
        weighted_bits = bits * self._weights
        packed_ints = tf.reduce_sum(weighted_bits, axis=-1)
        packed_ints = tf.cast(packed_ints, dtype=self.pack_bits_dtype)
        return packed_ints


class Expectation(tf.keras.layers.Layer):
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
