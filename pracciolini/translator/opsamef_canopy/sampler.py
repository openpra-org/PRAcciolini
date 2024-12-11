import tensorflow as tf
import numpy as np

def pack_tensor_bits(bool_tensor: tf.Tensor, dtype: tf.DType = tf.uint8) -> tf.Tensor:
    """
    Packs a boolean tensor into a tensor of specified integer dtype by treating each row as a sequence of bits.

    Args:
        bool_tensor: A boolean tensor of shape (x, y).
        dtype: An integer TensorFlow data type (e.g., tf.uint8).

    Returns:
        A tensor of shape (x, ceil(y / n_bits)) with dtype `dtype`, where n_bits is the bit width of `dtype`.
    """

    valid_dtypes = [tf.uint8, tf.int8, tf.int16, tf.uint16, tf.int32, tf.uint32, tf.int64, tf.uint64]

    if dtype not in valid_dtypes:
        raise tf.errors.InvalidArgumentError(dtype, dtype, f"requested dtype needs to be one of {valid_dtypes}")

    if bool_tensor.dtype is not tf.bool:
        raise tf.errors.InvalidArgumentError(bool_tensor, dtype, "input tensor needs to be a bool tensor!")

    # Get the number of bits in the dtype (e.g., 8 for tf.uint8)
    n_bits = tf.dtypes.as_dtype(dtype).size * 8

    if n_bits > tf.dtypes.as_dtype(tf.uint64).size * 8:
        raise tf.errors.InvalidArgumentError("Cannot handle word-size larger than `tf.uint64`")

    compute_dtype = tf.uint64

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

    # Generate weights for each bit position starting from the MSB
    weights = tf.constant([1 << (n_bits - 1 - i) for i in range(n_bits)], dtype=compute_dtype)

    # Convert bits to integers and apply weights
    bits = tf.cast(reshaped_tensor, dtype=compute_dtype)
    weighted_bits = bits * weights

    # Sum over the bits to get packed integers
    packed_ints = tf.reduce_sum(weighted_bits, axis=-1)

    # Cast to the desired dtype
    packed_tensor = tf.cast(packed_ints, dtype=dtype)

    return packed_tensor

def tensor_as_bit_vectors(tensor):
    """
    Accepts a tensor and prints its elements as bit-vectors.

    Args:
        tensor: A TensorFlow tensor.
    """

    # Get the dtype of the tensor
    dtype = tensor.dtype

    # Determine the bit-width of the data type
    bit_width = tf.dtypes.as_dtype(dtype).size * 8

    # Convert tensor to numpy array
    numpy_array = tensor.numpy()

    # Create a vectorized function to format each element
    vectorized_format = np.vectorize(lambda x: f'0b{format(x, f"0{bit_width}b")}')

    # Apply the vectorized function to the numpy array
    return vectorized_format(numpy_array)
