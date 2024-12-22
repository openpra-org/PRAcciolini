import tensorflow as tf

def my_bitwise_or(a, b):
    print(f"my_bitwise_or_shapes: a:{a.shape}, b:{b.shape}")
    bitwise_or_output = tf.bitwise.bitwise_or(a, b)
    print(f"my_bitwise_or_output_shape: {bitwise_or_output.shape}")
    return bitwise_or_output

#@tf.function(jit_compile=True)
def bitwise_nary_op(bitwise_op, inputs, axis=1):
    """
    Efficiently applies the n-ary bitwise op across the specified axis.

    Args:
        bitwise_op (function): The tf.experimental.numpy bitwise function to be applied.
        inputs (tf.Tensor): Input tensor with shape (..., num_events, ...).
        axis (int): The axis along which to apply the operation.

    Returns:
        tf.Tensor: Output tensor with reduced dimension along the specified axis.
    """
    print(f"input_shape: {inputs.shape}")
    # Transpose the inputs to have shape (num_events, batch_size)
    transposed_inputs = tf.transpose(inputs, perm=[1, 0, 2])
    print(f"transposed_inputs: {transposed_inputs.shape}")
    initial_value = tf.zeros(shape=(inputs.shape[0], inputs.shape[2]), dtype=inputs.dtype)
    print(f"initializer_shape: {initial_value.shape}")

    # Use tf.scan to apply bitwise OP across the num_events dimension
    result = tf.scan(
        fn=lambda a, b: bitwise_op(a, b),
        elems=transposed_inputs,
        initializer=initial_value,
        infer_shape=True, # potential runtime performance implications
        parallel_iterations=1024,
        reverse=False, #can potentially have performance implications for subsequent scans due locality
    )
    print(f"result_shape: {result.shape}")
    return result

@tf.function(jit_compile=True)
def bitwise_and(inputs, axis=1):
    return bitwise_nary_op(tf.bitwise.bitwise_and, inputs, axis=axis)

@tf.function(jit_compile=True)
def bitwise_or(inputs, axis=1):
    return bitwise_nary_op(my_bitwise_or, inputs, axis=axis)

@tf.function(jit_compile=True)
def bitwise_xor(inputs, axis=1):
    return bitwise_nary_op(tf.bitwise.bitwise_xor, inputs, axis=axis)

@tf.function(jit_compile=True)
def bitwise_not(inputs):
    return tf.bitwise.invert(inputs)

@tf.function(jit_compile=True)
def bitwise_nand(inputs, axis=1):
    return bitwise_not(bitwise_and(inputs, axis=axis))

@tf.function(jit_compile=True)
def bitwise_nor(inputs, axis=1):
    return bitwise_not(bitwise_or(inputs, axis=axis))

@tf.function(jit_compile=True)
def bitwise_xnor(inputs, axis=1):
    return bitwise_not(bitwise_xor(inputs, axis=axis))