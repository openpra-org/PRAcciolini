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
        # iterations, batch_size = compute_batching_parameters(num_events=num_events,
        #                                                      samples_per_event=samples_per_event,
        #                                                      sampling_precision=sampler_dtype,
        #                                                      bitpack_dtype=dtype)

        dtype = tf.uint8
        sampler_dtype = tf.float64
        num_events = 1024
        samples_per_event = 2 ** 20 ## ~ 1 million samples per basic event, 1 billion samples to collect overall
        batch_size = 2 ** 16
        iterations = samples_per_event // batch_size # ~67 million samples per batch

        stub_input = tf.keras.Input(shape=(1,), dtype=tf.uint64)
        samples = BitpackedBernoulli(
            name=f"p_samples",
            probs=[float(1.0 / (x + 1.0)) for x in range(num_events)],
            batch_size=batch_size,
            dtype=dtype,
            sampler_dtype=sampler_dtype)(stub_input)

        outputs = [samples]  # Extract samples directly as output
        model = tf.keras.Model(inputs=stub_input, outputs=outputs)
        model.compile()
        model.summary()

        # ---- Generate and Save Samples ---- #
        output_file_path = '/tmp/samples.h5'  # Output file path
        generate_and_save_samples(model, output_file_path, iterations, batch_size)

    def test_stream_samples_from_disk(self):
        output_file_path = '/tmp/samples.h5'  # Output file path
        total_samples = 2 ** 20
        batch_size = 2 ** 16
        dataset: tf.data.Dataset = create_dataset_from_hdf5(output_file_path).take(total_samples).batch(batch_size=batch_size, num_parallel_calls=128)
        # Example usage of the dataset
        iteration = 0

        for batch in dataset:
            iteration += 1
            print(f"iteration: {iteration} batch_items: {batch.shape[0]}, num_basic_events: {batch.shape[1]}")
            #print(batch)

if __name__ == '__main__':
    unittest.main()