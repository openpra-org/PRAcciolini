import tensorflow as tf


def pack_tensor_bits(bool_tensor: tf.Tensor, dtype: tf.DType = tf.uint8) -> tf.Tensor:
    """
    Packs a boolean tensor into a tensor of specified integer dtype by treating each row as a sequence of bits.

    Args:
        bool_tensor: A boolean tensor of shape (x, y).
        dtype: An integer TensorFlow data type (e.g., tf.uint8).

    Returns:
        A tensor of shape (x, ceil(y / n_bits)) with dtype `dtype`, where n_bits is the bit width of `dtype`.
    """

    valid_dtypes = [
        tf.int8,  tf.uint8,
        tf.int16, tf.uint16,
        tf.int32, tf.uint32,
        tf.int64, tf.uint64
    ]

    if dtype not in valid_dtypes:
        raise ValueError(f"Requested dtype must be one of {valid_dtypes}")

    if bool_tensor.dtype != tf.bool:
        raise ValueError("Input tensor must be of dtype tf.bool")

    # Get the number of bits in the dtype (e.g., 8 for tf.uint8)
    n_bits = tf.dtypes.as_dtype(dtype).size * 8

    if n_bits > tf.dtypes.as_dtype(tf.uint64).size * 8:
        raise ValueError("Cannot handle word-size larger than tf.uint64")

    # Precompute weights outside the @tf.function
    weights_list = [1 << (n_bits - 1 - i) for i in range(n_bits)]
    weights = tf.constant(weights_list, dtype=tf.uint64)

    # Call the inner function
    packed_tensor = _pack_tensor_bits_impl(bool_tensor=bool_tensor, weights=weights, dtype=dtype)

    return packed_tensor

@tf.function
def _pack_tensor_bits_impl(bool_tensor: tf.Tensor, weights: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    """
    Internal function to pack bits using TensorFlow operations.

    Args:
        bool_tensor: A boolean tensor.
        weights: A tensor containing weights for each bit position.

    Returns:
        A tensor containing packed integers.
    """

    n_bits = tf.shape(weights)[0]

    # Get the shape of the input tensor
    shape = tf.shape(bool_tensor)
    x = shape[0]  # Number of rows
    y = shape[1]  # Number of columns (bits per row)

    # Compute the total number of bits after padding
    y_padded = ((y + n_bits - 1) // n_bits) * n_bits  # Round up to the nearest multiple of n_bits
    pad_size = y_padded - y  # Number of bits to pad

    # Pad the boolean tensor on the right with False (equivalent to 0)
    padded_tensor = tf.pad(bool_tensor, [[0, 0], [0, pad_size]], constant_values=False)

    # Reshape to (x, num_packed_elements_per_row, n_bits)
    reshaped_tensor = tf.reshape(padded_tensor, [x, -1, n_bits])

    # Convert bits to integers and apply weights
    bits = tf.cast(reshaped_tensor, dtype=weights.dtype)
    weighted_bits = bits * weights

    # Sum over the bits to get packed integers
    packed_ints = tf.reduce_sum(weighted_bits, axis=-1)

    packed_tensor = tf.cast(packed_ints, dtype=dtype)

    return packed_tensor
