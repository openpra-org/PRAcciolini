import tensorflow as tf
from tensorflow.keras.layers import Layer
from functools import reduce

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
            raise ValueError("BitwiseAnd Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseAnd Layer requires at least two input tensors.")
        return reduce(tf.bitwise.bitwise_and, inputs)

class BitwiseOr(Layer):
    def __init__(self, **kwargs):
        super(BitwiseOr, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseOr Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseOr Layer requires at least two input tensors.")
        return reduce(tf.bitwise.bitwise_or, inputs)

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
