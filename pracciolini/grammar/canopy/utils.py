import tensorflow as tf
import numpy as np


def tensor_as_formatted_bit_vectors(tensor: tf.Tensor):
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
    formatted_array = vectorized_format(numpy_array)

    return formatted_array