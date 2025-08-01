import tensorflow as tf

#@tf.function(jit_compile=True)
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

def bitwise_nary_op3(bitwise_op, inputs, name: str):
    """
    Efficiently applies the n-ary bitwise op across the specified axis using barrier semantics.

    Args:
        bitwise_op (function): The bitwise reduction over the input tensor across the num_events dimension.
                               Can be one of `tf.bitwise.bitwise_or`, `tf.bitwise.bitwise_and`, `tf.bitwise.bitwise_xor`.
        inputs (tf.Tensor): Input tensor with shape (num_events, batch_size, sample_size).
        name (str): A name for this operation.

    Returns:
        tf.Tensor: Output tensor with shape [batch_size, sample_size] after reducing the num_events dimension via bitwise op.
    """
    # Ensure that we are in graph mode as tf.raw_ops.Barrier is not compatible with eager execution
    tf.compat.v1.disable_eager_execution()

    num_events = inputs.shape[0]
    batch_size = inputs.shape[1]
    sample_size = inputs.shape[2]

    # Create a barrier with the appropriate component types and shapes
    barrier_handle = tf.raw_ops.Barrier(
        component_types=[inputs.dtype],
        shapes=[inputs.shape[1:]],  # Shape of each value to insert
        capacity=-1,  # Unlimited capacity
        container='',
        shared_name=name + '_barrier'
    )

    # Insert each slice of the input tensor into the barrier
    insert_ops = []
    for i in tf.range(num_events):
        key = tf.strings.as_string(i)  # Generate a unique key for each slice
        keys = tf.expand_dims(key, axis=0)  # Shape [1]
        values = [tf.expand_dims(inputs[i], axis=0)]  # Shape [1, batch_size, sample_size]

        insert_op = tf.raw_ops.BarrierInsertMany(
            handle=barrier_handle,
            keys=keys,
            values=values,
            component_index=0  # Added this line
        )
        insert_ops.append(insert_op)

    # Ensure all insertions are completed before proceeding
    with tf.control_dependencies(insert_ops):
        # Retrieve all the values from the barrier
        outputs = tf.raw_ops.BarrierTakeMany(
            handle=barrier_handle,
            num_elements=num_events,
            component_types=[inputs.dtype],  # Added this line
            allow_small_batch=True,
            wait_for_incomplete=False  # All elements have been inserted
        )

        # Extract the values from the outputs
        # outputs[2] is a list of tensors; we extract the first (and only) component
        values_out = outputs[2][0]  # Shape: [num_events, batch_size, sample_size]

    # Perform the reduction across the num_events dimension using a while loop
    i = tf.constant(1)
    result = values_out[0]

    def condition(i, result):
        return tf.less(i, num_events)

    def body(i, result):
        result = bitwise_op(result, values_out[i])
        i = tf.add(i, 1)
        return i, result

    # Execute the while loop to reduce the values
    _, final_result = tf.while_loop(condition, body, [i, result])

    return final_result


def efficient_bitwise_op(op, inputs, name="no_fold_op"):
    """
    Efficiently computes the bitwise OR across the first dimension of the inputs.

    Args:
        inputs (tf.Tensor): Input tensor with shape (num_events, batch_size, sample_size).
                            Should be of integer dtype (e.g., tf.uint8).

    Returns:
        tf.Tensor: Output tensor with shape (batch_size, sample_size) after computing
                   the bitwise OR across the num_events dimension.
    """
    # Expand bits to shape: (num_events, batch_size, sample_size, bit_width)
    bit_width = inputs.dtype.size * 8
    bit_range = tf.cast(tf.range(bit_width), dtype=inputs.dtype)
    bits = tf.bitwise.right_shift(tf.expand_dims(inputs, axis=-1), bit_range) & 1

    # Perform bitwise OR across the num_events dimension
    bits_op = op(bits, axis=0)  # Shape: (batch_size, sample_size, bit_width)

    # Reconstruct the numbers from bits
    shifted_bits = tf.bitwise.left_shift(bits_op, bit_range)
    result = tf.reduce_sum(shifted_bits, axis=-1)
    result = tf.cast(result, inputs.dtype, name=name)
    return result


def efficient_bitwise_or(inputs, name="no_fold_or"):
    return efficient_bitwise_op(tf.reduce_max, inputs, name=name)

def efficient_bitwise_and(inputs, name="no_fold_and"):
    """
    Efficiently computes the bitwise AND across the first dimension of the inputs.

    Args:
        inputs (tf.Tensor): Input tensor with shape (num_events, batch_size, sample_size).
                            Should be of integer dtype (e.g., tf.uint8, tf.uint16).

    Returns:
        tf.Tensor: Output tensor with shape (batch_size, sample_size) after computing
                   the bitwise AND across the num_events dimension.
    """
    # Perform bitwise AND using reduce_min
    return efficient_bitwise_op(tf.reduce_min, inputs, name=name)

# @tf.function(jit_compile=True)
def bitwise_nary_op(bitwise_op, inputs, name: str):
    """
    Efficiently applies the n-ary bitwise op across the specified axis.

    Args:
        bitwise_op (function): The bitwise reduction over the input tensor across the num_events dimension. can be one of `tf.bitwise.bitwise_or`, tf.bitwise.bitwise_and`, `tf.bitwise.bitwise_xor`
        inputs (tf.Tensor): Input tensor with shape (num_events, batch_size, sample_size).
        name (str): A name for this operation

    Returns:
        tf.Tensor: Output tensor with shape [batch_size, sample_size] after reducing the num_events dimension via bitwise op.
    """
    #print("inputs_shape:", inputs)
    #accumulator_ = tf.zeros([inputs.shape[1], inputs.shape[2]], dtype=inputs.dtype)
    result = tf.foldl(
        fn=bitwise_op,
        elems=inputs,
        initializer=inputs[0, :, :],
        parallel_iterations=(inputs.shape[0] + 1),
        swap_memory=True,
        name=name,
    )
    return result

#@tf.function(jit_compile=True)
def bitwise_atleast(inputs, name: str = "atleast", atleast = 1):
    return bitwise_nary_op(tf.bitwise.bitwise_or, inputs, name=name)

#@tf.function(jit_compile=True)
def bitwise_and(inputs, name: str = "and"):
    return bitwise_nary_op(bitwise_op=tf.bitwise.bitwise_and, inputs=inputs, name=name)

#@tf.function(jit_compile=True)
def bitwise_or(inputs, name: str = "or"):
    return bitwise_nary_op(bitwise_op=tf.bitwise.bitwise_or, inputs=inputs, name=name)

#@tf.function(jit_compile=True)
def bitwise_xor(inputs, name: str = "xor"):
    return bitwise_nary_op(bitwise_op=tf.bitwise.bitwise_xor, inputs=inputs, name=name)

#@tf.function(jit_compile=True)
def bitwise_not(inputs, name: str = "not"):
    return tf.bitwise.invert(x=inputs, name=name)

#@tf.function(jit_compile=True)
def bitwise_nand(inputs, name: str = "nand"):
    return bitwise_not(bitwise_and(inputs=inputs,name=name))

#@tf.function(jit_compile=True)
def bitwise_nor(inputs, name: str = "nor"):
    return bitwise_not(bitwise_or(inputs=inputs,name=name))

#@tf.function(jit_compile=True)
def bitwise_xnor(inputs, name: str = "xnor"):
    return bitwise_not(bitwise_xor(inputs=inputs,name=name))