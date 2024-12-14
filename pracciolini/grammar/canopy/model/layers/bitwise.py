import tensorflow as tf


class BitwiseNot(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BitwiseNot, self).__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError("BitwiseNot Layer requires exactly one input tensor.")
            inputs = inputs[0]
        return self.compute_bitwise_not(inputs)

    @tf.function
    def compute_bitwise_not(self, inputs):
        return tf.bitwise.invert(inputs)


class BitwiseAnd(tf.keras.layers.Layer):
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
        return tf.foldl(lambda a, b: tf.bitwise.bitwise_and(a, b), stacked_inputs)


class BitwiseOr(tf.keras.layers.Layer):
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
        stacked_inputs = tf.stack(inputs)
        return tf.foldl(lambda a, b: tf.bitwise.bitwise_or(a, b), stacked_inputs)


class BitwiseXor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BitwiseXor, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseXor Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseXor Layer requires at least two input tensors.")
        return self.compute_bitwise_xor(inputs)

    @tf.function
    def compute_bitwise_xor(self, inputs):
        stacked_inputs = tf.stack(inputs)
        result = tf.foldl(lambda a, b: tf.bitwise.bitwise_xor(a, b), stacked_inputs)
        return result


class BitwiseNand(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BitwiseNand, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseNand Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseNand Layer requires at least two input tensors.")
        return self.compute_bitwise_nand(inputs)

    @tf.function
    def compute_bitwise_nand(self, inputs):
        stacked_inputs = tf.stack(inputs)
        result = tf.foldl(lambda a, b: tf.bitwise.bitwise_and(a, b), stacked_inputs)
        return tf.bitwise.invert(result)


class BitwiseNor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BitwiseNor, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseNor Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseNor Layer requires at least two input tensors.")
        return self.compute_bitwise_nor(inputs)

    @tf.function
    def compute_bitwise_nor(self, inputs):
        stacked_inputs = tf.stack(inputs)
        result = tf.foldl(lambda a, b: tf.bitwise.bitwise_or(a, b), stacked_inputs)
        return tf.bitwise.invert(result)


class BitwiseXnor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BitwiseXnor, self).__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("BitwiseXnor Layer requires a list of input tensors.")
        if len(inputs) < 2:
            raise ValueError("BitwiseXnor Layer requires at least two input tensors.")
        return self.compute_bitwise_xnor(inputs)

    @tf.function
    def compute_bitwise_xnor(self, inputs):
        stacked_inputs = tf.stack(inputs)
        result = tf.foldl(lambda a, b: tf.bitwise.bitwise_xor(a, b), stacked_inputs)
        return tf.bitwise.invert(result)


class BitwiseKofN(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(BitwiseKofN, self).__init__(**kwargs)

    def call(self, inputs):
        return self.compute_bitwise_k_of_n(inputs)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.uint8)]
    )
    def compute_bitwise_k_of_n(self, inputs):
        # Inputs: Tensor of arbitrary shape with dtype tf.uint8
        # Goal: Compute per-bit-position counts of bits set to 1 across all inputs
        #       and determine if the counts equal 'k' at each bit position

        # Step 1: Expand the input tensor to add an extra dimension for bit positions
        # This allows us to perform broadcasting over the bit positions
        # Shape after expansion: [..., 1]
        bits = tf.expand_dims(inputs, axis=-1)

        # Step 2: Create a tensor representing the bit positions (0 to 7 for 8 bits)
        # Using tf.range to create [0, 1, 2, 3, 4, 5, 6, 7]
        bit_positions = tf.range(8, dtype=tf.uint8)

        # Step 3: Use tf.raw_ops.RightShift to shift bits accordingly
        # This operation shifts each element in 'bits' right by the corresponding bit position
        # Broadcasting occurs over the newly added dimension for bit positions
        shifted_bits = tf.raw_ops.RightShift(x=bits, y=bit_positions)

        # Step 4: Use tf.raw_ops.BitwiseAnd to extract the least significant bit (LSB)
        # This effectively gives us the bit at each position for each element in 'inputs'
        # The result is a tensor of 0s and 1s indicating the bit value at each position
        bits_extracted = tf.raw_ops.BitwiseAnd(x=shifted_bits, y=1)

        # Step 5: Convert the bits to tf.int32 for summation
        bits_extracted_int32 = tf.cast(bits_extracted, tf.int32)

        # Step 6: Sum over all dimensions except the last one (which represents bit positions)
        # This computes the count of bits set to 1 at each bit position across all inputs
        axes_to_sum_over = tf.range(tf.rank(bits_extracted_int32) - 1)
        counts_per_bit_position = tf.reduce_sum(bits_extracted_int32, axis=axes_to_sum_over)

        # Step 7: Compare the counts with 'k' to determine if they match
        # The result is a tensor of booleans indicating where the counts equal 'k'
        k_equals_counts = tf.equal(counts_per_bit_position, self.k)

        # Step 8: Cast the boolean results back to the inputs' dtype (tf.uint8)
        # This results in a tensor of 0s and 1s indicating where counts equal 'k'
        result = tf.cast(k_equals_counts, inputs.dtype)

        return result
