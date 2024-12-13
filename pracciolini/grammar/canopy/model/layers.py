import tensorflow as tf
from tensorflow.keras.layers import Layer
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
        # Use tf.foldl to perform the bitwise AND reduction efficiently.
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
