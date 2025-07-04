import tensorflow as tf
from tensorflow.keras.layers import Layer

from pracciolini.grammar.canopy.model.ops.bitwise import bitwise_nary_op, bitwise_not, bitwise_or, bitwise_xor, bitwise_nand, \
    bitwise_nor, bitwise_xnor
from pracciolini.grammar.canopy.model.ops.monte_carlo import tally
from pracciolini.grammar.canopy.model.ops.sampler import generate_bernoulli


class Expectation(Layer):
    """
    Custom Keras Layer that computes the expected value (mean) of bits set to 1 in the input tensor.

    This layer wraps the 'expectation_impl' function to be used within a Keras model.
    """

    def __init__(self, **kwargs):
        super(Expectation, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(Expectation, self).__call__(*args, **kwargs)

    def call(self, inputs):
        """
        Invokes the layer on the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after computing the expected value.
        """
        return tally(inputs)


class BitwiseNot(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNot, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseNot, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_not(inputs)


class BitwiseAnd(Layer):
    def __init__(self, **kwargs):
        super(BitwiseAnd, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseAnd, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_nary_op(tf.bitwise.bitwise_and, inputs)


class BitwiseOr(Layer):
    def __init__(self, **kwargs):
        super(BitwiseOr, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseOr, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_or(inputs)


class BitwiseXor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXor, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseXor, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_xor(inputs)


class BitwiseNand(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNand, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseNand, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_nand(inputs)


class BitwiseNor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNor, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseNor, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_nor(inputs)


class BitwiseXnor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXnor, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return super(BitwiseXnor, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return bitwise_xnor(inputs)

class BitpackedBernoulli(Layer):
    def __init__(self, sample_size, bitpack_dtype=tf.uint8, dtype=tf.float32, **kwargs):
        super(BitpackedBernoulli, self).__init__(**kwargs)
        self._sample_size = sample_size
        self._bitpack_dtype = bitpack_dtype
        self._dtype = dtype

    def __call__(self, *args, **kwargs):
        return super(BitpackedBernoulli, self).__call__(*args, **kwargs)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return generate_bernoulli(
            probs=inputs,
            n_sample_packs_per_probability=self._sample_size,
            bitpack_dtype=self._bitpack_dtype,
            dtype=self._dtype,
            seed=None
        )