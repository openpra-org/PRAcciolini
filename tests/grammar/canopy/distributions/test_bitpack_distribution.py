import unittest

import tensorflow as tf

from pracciolini.grammar.canopy.probability.distributions import Bernoulli, Binomial, Categorical


class BitpackDistributionMixinTests(unittest.TestCase):

    def test_construct_no_bitpack_bool_bernoulli(self):
        dist = Bernoulli(probs=0.5, dtype=tf.bool)
        num_bits_list = [1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]
        for num_bits in num_bits_list:
            with self.subTest(num_bits=num_bits):
                samples = dist.sample(sample_shape=(num_bits,), seed=372, pack_bits=None)
                # Unpacked samples, shape should be (num_bits,)
                expected_shape = (num_bits,)
                self.assertEqual(samples.shape, expected_shape)
                self.assertEqual(samples.dtype, dist.dtype)

    def test_construct_bitpack_bool_bernoulli(self):
        dist = Bernoulli(probs=0.5, dtype=tf.bool)
        num_bits_list = [1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]
        dtypes = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
        for dtype in dtypes:
            for num_bits in num_bits_list:
                with self.subTest(dtype=dtype, num_bits=num_bits):
                    # Sample with specified sample_shape
                    samples = dist.sample(sample_shape=(num_bits,), seed=372, pack_bits=dtype)
                    # Packed samples, shape depends on n_bits and dtype
                    bits_per_word = tf.dtypes.as_dtype(dtype).size * 8
                    num_packed_elements = (num_bits + bits_per_word - 1) // bits_per_word
                    expected_shape = (num_packed_elements,)
                    self.assertEqual(samples.shape, expected_shape)
                    self.assertEqual(samples.dtype, dtype)

    def test_bernoulli_different_dtypes(self):
        # Test Bernoulli with different event dtypes
        probs = [0.5, 0.8]
        dtypes = [tf.bool, tf.int16, tf.int32, tf.float32, tf.float64]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                dist = Bernoulli(probs=probs, dtype=dtype)
                samples = dist.sample(sample_shape=(10,), seed=42)
                expected_dtype = dtype
                self.assertEqual(samples.dtype, expected_dtype)
                # Check that samples are in the expected range
                samples_np = samples.numpy()
                if dtype == tf.bool:
                    self.assertTrue(
                        ((samples_np == True) | (samples_np == False)).all()
                    )
                elif dtype.is_integer:
                    self.assertTrue(
                        ((samples_np == 0) | (samples_np == 1)).all()
                    )
                elif dtype.is_floating:
                    self.assertTrue(
                        ((samples_np == 0.0) | (samples_np == 1.0)).all()
                    )

    def test_bitpack_bernoulli_batch(self):
        # Test Bernoulli with batch shape
        probs = [0.2, 0.5, 0.8]
        dist = Bernoulli(probs=probs, dtype=tf.bool)
        sample_shape = (10,)
        dtype = tf.uint8

        samples = dist.sample(sample_shape=sample_shape, pack_bits=dtype)
        # The expected shape is batch_shape + [num_packed_elements_per_row]
        bits_per_word = tf.dtypes.as_dtype(dtype).size * 8
        num_bits = sample_shape[0]
        num_packed_elements = (num_bits + bits_per_word - 1) // bits_per_word

        expected_shape = dist.batch_shape + (num_packed_elements,)
        self.assertEqual(samples.shape, expected_shape)
        self.assertEqual(samples.dtype, dtype)

    def test_construct_bitpack_categorical(self):
        dist = Categorical(logits=[0.1, 0.2, 0.7], dtype=tf.int32)

        dtypes = [None, tf.uint8]
        num_samples_list = [5, 10]

        for dtype in dtypes:
            for num_samples in num_samples_list:
                with self.subTest(dtype=dtype, num_samples=num_samples):
                    samples = dist.sample(sample_shape=(num_samples,), seed=42, pack_bits=dtype)
                    if dtype is None:
                        expected_shape = (num_samples,)
                        self.assertEqual(samples.shape, expected_shape)
                        self.assertEqual(samples.dtype, dist.dtype)
                        samples_np = samples.numpy()
                        self.assertTrue(
                            ((samples_np >= 0) & (samples_np < 3)).all()
                        )
                    else:
                        # Casting samples to bool will result in True for non-zero class indices
                        # and False for class index 0
                        bits_per_word = tf.dtypes.as_dtype(dtype).size * 8
                        num_packed_elements = (num_samples + bits_per_word - 1) // bits_per_word
                        expected_shape = (num_packed_elements,)
                        self.assertEqual(samples.shape, expected_shape)