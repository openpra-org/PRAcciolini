import tensorflow as tf

@tf.function(jit_compile=True)
def bitwise_nary_op2(bitwise_op, inputs):
    """
    Constructs a bitwise XOR reduction over the input tensor across the num_events dimension.
    Args:
        inputs (tf.Tensor): Input tensor with shape [num_events, batch_size, sample_size] and dtype tf.uint8.
    Returns:
        tf.Tensor: Output tensor with shape [batch_size, sample_size] after reducing the num_events dimension via bitwise op.
    """
    # Ensure that num_events is known at compile time
    batch_size = inputs.shape[1]
    sample_size = inputs.shape[2]


    # Transpose inputs to have num_events as the first dimension
    # inputs_T = tf.transpose(inputs, perm=[1, 0, 2])  # Shape: [num_events, batch_size, sample_size]

    # Use tf.scan to perform cumulative bitwise XOR over the num_events dimension
    # tf.scan applies the XOR function cumulatively and returns the final result
    def xor_fn(accumulator, current):
        return tf.bitwise.bitwise_xor(accumulator, current)


    # Initialize the accumulator with zeros
    initial_accumulator = tf.zeros([batch_size, sample_size], dtype=inputs.dtype)

    # Perform the reduction using tf.scan or tf.foldl
    output = tf.foldl(
        xor_fn,
        elems=inputs,
        parallel_iterations=16,
        initializer=initial_accumulator,
    )  # Output shape: [batch_size, sample_size]

    # Alternatively, since we're only interested in the final result, you can use tf.reduce if available
    # output = tf.experimental.numpy.bitwise_xor.reduce(inputs, axis=1)

    return output  # Shape: [batch_size, sample_size]


@tf.function(jit_compile=True)
def bitwise_nary_op(bitwise_op, inputs):
    """
    Efficiently applies the n-ary bitwise op across the specified axis.

    Args:
        bitwise_op (function): The bitwise reduction over the input tensor across the num_events dimension.
        inputs (tf.Tensor): Input tensor with shape (num_events, batch_size, sample_size).

    Returns:
        tf.Tensor: Output tensor with shape [batch_size, sample_size] after reducing the num_events dimension via bitwise op.
    """
    accumulator_ = tf.zeros([inputs.shape[1], inputs.shape[2]], dtype=inputs.dtype)
    result = tf.foldl(
        fn=bitwise_op,
        elems=inputs,
        initializer=accumulator_,
        parallel_iterations=32,
        swap_memory=True,
    )
    return result

@tf.function(jit_compile=True)
def bitwise_and(inputs):
    return bitwise_nary_op(tf.bitwise.bitwise_and, inputs)

@tf.function(jit_compile=True)
def bitwise_or(inputs):
    return bitwise_nary_op(tf.bitwise.bitwise_or, inputs)

@tf.function(jit_compile=True)
def bitwise_xor(inputs):
    return bitwise_nary_op(tf.bitwise.bitwise_xor, inputs)

@tf.function(jit_compile=True)
def bitwise_not(inputs):
    return tf.bitwise.invert(inputs)

@tf.function(jit_compile=True)
def bitwise_nand(inputs):
    return bitwise_not(bitwise_and(inputs))

@tf.function(jit_compile=True)
def bitwise_nor(inputs):
    return bitwise_not(bitwise_or(inputs))

@tf.function(jit_compile=True)
def bitwise_xnor(inputs):
    return bitwise_not(bitwise_xor(inputs))