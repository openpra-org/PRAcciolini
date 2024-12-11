import tensorflow as tf
from tensorflow.keras.layers import Layer

class BitwiseNot(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNot, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.bitwise.invert(inputs)

class BitwiseAnd(Layer):
    def __init__(self, **kwargs):
        super(BitwiseAnd, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.bitwise.bitwise_and(x, y)

class BitwiseOr(Layer):
    def __init__(self, **kwargs):
        super(BitwiseOr, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.bitwise.bitwise_or(x, y)

class BitwiseXor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXor, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.bitwise.bitwise_xor(x, y)

# Additional layers for BITWISE_NAND, BITWISE_NOR, BITWISE_XNOR
class BitwiseNand(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNand, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.bitwise.invert(tf.bitwise.bitwise_and(x, y))

class BitwiseNor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseNor, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.bitwise.invert(tf.bitwise.bitwise_or(x, y))

class BitwiseXnor(Layer):
    def __init__(self, **kwargs):
        super(BitwiseXnor, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.bitwise.invert(tf.bitwise.bitwise_xor(x, y))