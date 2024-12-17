import tensorflow as tf


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
    print(inputs)
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

@tf.function
def bitwise_and(inputs):
    return bitwise_nary_op(tf.bitwise.bitwise_and, inputs)

@tf.function
def bitwise_or(inputs):
    return bitwise_nary_op(tf.bitwise.bitwise_or, inputs)

@tf.function
def bitwise_xor(inputs):
    return bitwise_nary_op(tf.bitwise.bitwise_xor, inputs)

@tf.function
def bitwise_not(inputs):
    return tf.bitwise.invert(inputs)

@tf.function
def bitwise_nand(inputs):
    return bitwise_not(bitwise_and(inputs))

@tf.function
def bitwise_nor(inputs):
    return bitwise_not(bitwise_or(inputs))

@tf.function
def bitwise_xnor(inputs):
    return bitwise_not(bitwise_xor(inputs))