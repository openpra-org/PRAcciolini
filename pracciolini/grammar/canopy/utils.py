from typing import Optional, Tuple

import tensorflow as tf
import numpy as np
import h5py

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




def generate_and_save_samples(model, output_path, num_iterations, batch_size):
    """
    Generates samples using the model and saves them to disk.

    Args:
        model (tf.keras.Model): The Keras model that outputs the samples.
        output_path (str): Path to the HDF5 file where samples will be saved.
        num_iterations (int): Number of iterations to generate samples.
        batch_size (int): Batch size for sample generation.
    """
    total_batches = num_iterations
    sample_shape = model.output_shape  # Should be (batch_size, n_probs)

    # Create an HDF5 file to store the samples
    with h5py.File(output_path, 'w') as hf:
        # Create a dataset to store samples incrementally
        dset = hf.create_dataset('samples',
                                 shape=(0, sample_shape[-1]),
                                 maxshape=(None, sample_shape[-1]),
                                 chunks=True,
                                 dtype=np.uint8)

        # Generate and save samples in batches
        def run():
            input_data = tf.constant(0, shape=(1,))
            return model.predict(x=input_data)

        for i in range(total_batches):
            print(f"Generating batch {i+1}/{total_batches}")
            # Generate samples
            generated_samples = run()
            # Append samples to the dataset
            current_shape = dset.shape[0]
            new_shape = current_shape + generated_samples.shape[0]
            dset.resize(new_shape, axis=0)
            dset[current_shape:new_shape, :] = generated_samples.astype(np.uint8)


def create_dataset_from_hdf5(file_path, dataset_name='samples') -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset inside the HDF5 file.

    Returns:
        tf.data.Dataset: The dataset object for streaming samples.
    """

    def generator():
        with h5py.File(file_path, 'r') as hf:
            data = hf[dataset_name]
            for i in range(len(data)):
                yield data[i]

    with h5py.File(file_path, 'r') as hf:
        data_shape = hf[dataset_name].shape

    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=tf.uint8,
        output_shapes=(data_shape[1],)
    )

    #dataset = dataset.batch(batch_size)
    return dataset



def _compute_bits_in_dtype(tensor_type: tf.DType) -> int:
    """
    Computes the number of bits in the given data type.

    Args:
        tensor_type (tf.DType): The tensor data type.

    Returns:
        int: Number of bits in the data type.
    """
    return tf.dtypes.as_dtype(tensor_type).size * 8

def compute_optimal_sample_shape_for_constraints(
    num_events: int,
    max_bytes: Optional[int] = int(2 ** 32),
    dtype: Optional[tf.DType] = tf.float32,
    bitpack_dtype: Optional[tf.DType] = tf.uint8,
    batch_size_range: Optional[Tuple[Optional[int], Optional[int]]] = (1, None),
    sample_size_range: Optional[Tuple[Optional[int], Optional[int]]] = (int(2 ** 10), None),
) -> Tuple[int, int, dict]:
    """
    Computes the optimal sample shape (batch_size, sample_size) within the given constraints.

    Args:
        num_events (int): Number of events/probabilities.
        max_bytes (Optional[int], optional): Maximum allowed bytes for memory usage. Defaults to 4 GiB.
        dtype (Optional[tf.DType], optional): Data type for sampling. Defaults to tf.float32.
        bitpack_dtype (Optional[tf.DType], optional): Data type for bit-packing. Defaults to tf.uint8.
        batch_size_range (Optional[Tuple[Optional[int], Optional[int]]], optional):
            Minimum and maximum batch sizes. Defaults to (1, None).
        sample_size_range (Optional[Tuple[Optional[int], Optional[int]]], optional):
            Minimum and maximum sample sizes (n_sample_packs_per_probability). Defaults to (1024, None).

    Returns:
        Tuple[int, int, dict]: Optimal (batch_size, sample_size) that fit within the constraints,
                               and a dictionary of internal variables for debugging.

    Raises:
        ValueError: If constraints cannot be satisfied with given ranges.
    """
    debug_info = {}  # Dictionary to hold all internal variables

    # Compute bits per pack and size of dtype in bytes
    bits_in_bitpack_dtype = _compute_bits_in_dtype(bitpack_dtype)  # Bits per pack
    size_of_dtype_in_bytes = tf.dtypes.as_dtype(dtype).size        # Bytes per element

    debug_info['bits_in_bitpack_dtype'] = bits_in_bitpack_dtype
    debug_info['size_of_dtype_in_bytes'] = size_of_dtype_in_bytes

    # Unpack batch size and sample size ranges
    batch_size_min, batch_size_max = batch_size_range
    sample_size_min, sample_size_max = sample_size_range

    # Set defaults for None values
    if batch_size_min is None:
        batch_size_min = 1
    if batch_size_max is None:
        batch_size_max = int(2 ** 31 - 1)  # Max int32 value
    if sample_size_min is None:
        sample_size_min = 1
    if sample_size_max is None:
        sample_size_max = int(2 ** 31 - 1)

    debug_info['batch_size_min'] = batch_size_min
    debug_info['batch_size_max'] = batch_size_max
    debug_info['sample_size_min'] = sample_size_min
    debug_info['sample_size_max'] = sample_size_max

    # Validate input ranges
    if batch_size_min > batch_size_max:
        debug_info['error'] = "batch_size_min cannot be greater than batch_size_max"
        raise ValueError(f"{debug_info}")
    if sample_size_min > sample_size_max:
        debug_info['error'] = "sample_size_min cannot be greater than sample_size_max"
        raise ValueError(f"{debug_info}")

    num_events = int(num_events)
    max_bytes = int(max_bytes)

    debug_info['num_events'] = num_events
    debug_info['max_bytes'] = max_bytes

    # Compute constants for memory calculation
    bytes_per_sample = num_events * bits_in_bitpack_dtype * size_of_dtype_in_bytes * 2  # 2 for dist and samples tensors

    debug_info['bytes_per_sample'] = bytes_per_sample

    # Function to compute memory usage
    def compute_memory_usage(batch_size_, sample_size_):
        memory_usage = batch_size_ * sample_size_ * bytes_per_sample
        debug_info['current_memory_usage'] = memory_usage
        debug_info['current_batch_size'] = batch_size_
        debug_info['current_sample_size'] = sample_size_
        return memory_usage

    # Function to check if parameters satisfy constraints
    def is_within_constraints(batch_size_, sample_size_):
        memory_usage = compute_memory_usage(batch_size_, sample_size_)
        return memory_usage <= max_bytes

    # Attempt to maximize sample_size first
    # For the given batch_size_min, compute the maximal sample_size
    max_possible_sample_size = max_bytes // (batch_size_min * bytes_per_sample)
    max_possible_sample_size = min(max_possible_sample_size, sample_size_max)

    debug_info['max_possible_sample_size_initial'] = max_possible_sample_size

    # Check if max_possible_sample_size is within sample_size_range
    if max_possible_sample_size < sample_size_min:
        debug_info['error'] = "Cannot satisfy constraints with given batch_size_min and sample_size_range"
        raise ValueError(f"{debug_info}")

    # Try to find the optimal sample size within constraints
    if is_within_constraints(batch_size_min, max_possible_sample_size):
        optimal_sample_size = max_possible_sample_size
    else:
        # Reduce sample_size
        sample_size = max_possible_sample_size
        while sample_size >= sample_size_min:
            if is_within_constraints(batch_size_min, sample_size):
                optimal_sample_size = sample_size
                break
            sample_size -= 1
        else:
            debug_info['error'] = "Cannot satisfy constraints with given batch_size_min and sample_size_range"
            raise ValueError(f"{debug_info}")

    debug_info['optimal_sample_size'] = optimal_sample_size

    # Now, try to increase batch_size as much as possible
    max_possible_batch_size = max_bytes // (optimal_sample_size * bytes_per_sample)
    max_possible_batch_size = min(max_possible_batch_size, batch_size_max)

    debug_info['max_possible_batch_size_initial'] = max_possible_batch_size

    # Check if max_possible_batch_size is within batch_size_range
    if max_possible_batch_size < batch_size_min:
        debug_info['error'] = "Cannot satisfy constraints with given sample_size and batch_size_range"
        raise ValueError(f"{debug_info}")

    # Try to find the optimal batch size within constraints
    if is_within_constraints(max_possible_batch_size, optimal_sample_size):
        optimal_batch_size = max_possible_batch_size
    else:
        # Reduce batch_size
        batch_size = max_possible_batch_size
        while batch_size >= batch_size_min:
            if is_within_constraints(batch_size, optimal_sample_size):
                optimal_batch_size = batch_size
                break
            batch_size -= 1
        else:
            debug_info['error'] = "Cannot satisfy constraints with given sample_size and batch_size_range"
            raise ValueError(f"{debug_info}")

    debug_info['optimal_batch_size'] = optimal_batch_size

    # Final validation
    total_memory_bytes = compute_memory_usage(optimal_batch_size, optimal_sample_size)
    debug_info['total_memory_bytes'] = total_memory_bytes

    if total_memory_bytes > max_bytes:
        debug_info['error'] = "Cannot satisfy constraints with the given parameters"
        raise ValueError(f"{debug_info}")

    return int(optimal_batch_size), int(optimal_sample_size), debug_info