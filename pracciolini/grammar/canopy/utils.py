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


def create_dataset_from_hdf5(file_path, dataset_name='samples', batch_size=1024):
    """
    Creates a tf.data.Dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset inside the HDF5 file.
        batch_size (int): Batch size for the dataset.

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

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=tf.uint8,
        output_shapes=(data_shape[1],)
    )

    dataset = dataset.batch(batch_size)
    return dataset
