import unittest
import tensorflow as tf

from pracciolini.grammar.canopy.model.layers import BitpackedBernoulli, Expectation
from pracciolini.grammar.canopy.utils import generate_and_save_samples, create_dataset_from_hdf5

def compute_batching_parameters(num_events: int,
                                samples_per_event: int = int(2 ** 30), # approx. ~1 billion
                                sampling_precision: tf.DType = tf.float64,
                                bitpack_dtype: tf.DType = tf.uint8):
    sample_width = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    max_batch_size = int(float(tf.dtypes.int32.max) // float(sample_width * num_events))
    batch_size = max_batch_size if samples_per_event > max_batch_size else samples_per_event
    iterations = int(samples_per_event // batch_size)

    print(sample_width, batch_size, iterations)
    return iterations, batch_size

class SampleStreamingTests(unittest.TestCase):

    def test_stream_samples_to_disk(self):
        num_events = 128
        samples_per_event = 1024
        dtype = tf.uint8
        sampler_dtype = tf.float64
        iterations, batch_size = compute_batching_parameters(num_events=num_events,
                                                             samples_per_event=samples_per_event,
                                                             sampling_precision=sampler_dtype,
                                                             bitpack_dtype=dtype)
        stub_input = tf.keras.Input(shape=(1,), dtype=tf.uint64)
        samples = BitpackedBernoulli(
            name=f"p_samples",
            probs=[1.0 / (x + 1) for x in range(num_events)],
            batch_size=batch_size,
            dtype=dtype,
            sampler_dtype=sampler_dtype)(stub_input)

        outputs = [samples]  # Extract samples directly as output
        model = tf.keras.Model(inputs=stub_input, outputs=outputs)
        model.compile()
        model.summary()

        # ---- Generate and Save Samples ---- #
        output_file_path = 'samples.h5'  # Output file path
        generate_and_save_samples(model, output_file_path, iterations, batch_size)

    def test_stream_samples_from_disk(self):
        output_file_path = 'samples.h5'  # Output file path
        dataset = create_dataset_from_hdf5(output_file_path, batch_size=1024)

        # Example usage of the dataset
        for batch in dataset.take(1):
            print(f"Batch shape: {batch.shape}")
            print(batch)

if __name__ == '__main__':
    unittest.main()