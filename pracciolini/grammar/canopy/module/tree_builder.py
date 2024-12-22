import tensorflow as tf


@tf.function(jit_compile=True)
def build_binary_xor_tree(inputs):
    """
    Constructs a bitwise XOR reduction over the input tensor across the num_events dimension.
    Args:
        inputs (tf.Tensor): Input tensor with shape [batch_size, num_events, sample_size] and dtype tf.uint8.
    Returns:
        tf.Tensor: Output tensor with shape [batch_size, sample_size] after reducing the num_events dimension via bitwise XOR operations.
    """
    # Ensure that num_events is known at compile time
    batch_size = inputs.shape[0]
    num_events = inputs.shape[1]
    sample_size = inputs.shape[2]
    if any(dim is None for dim in [batch_size, num_events, sample_size]):
        raise ValueError("All input dimensions must be known at compile time for XLA compilation.")

    # Transpose inputs to have num_events as the first dimension
    inputs_T = tf.transpose(inputs, perm=[1, 0, 2])  # Shape: [num_events, batch_size, sample_size]

    # Use tf.scan to perform cumulative bitwise XOR over the num_events dimension
    # tf.scan applies the XOR function cumulatively and returns the final result
    def xor_fn(accumulator, current):
        return tf.bitwise.bitwise_xor(accumulator, current)

    # Initialize the accumulator with zeros
    initial_accumulator = tf.zeros([batch_size, sample_size], dtype=inputs.dtype)

    # Perform the reduction using tf.scan or tf.foldl
    output = tf.foldl(
        xor_fn,
        elems=inputs_T,
        initializer=initial_accumulator,
    )  # Output shape: [batch_size, sample_size]

    # Alternatively, since we're only interested in the final result, you can use tf.reduce if available
    # output = tf.experimental.numpy.bitwise_xor.reduce(inputs, axis=1)

    return output  # Shape: [batch_size, sample_size]