import unittest
import tensorflow as tf


class TestBitPackTensor(unittest.TestCase):
    def test_base_case_uint8(self):
        """
        Test the base case with a simple boolean tensor and dtype tf.uint8
        """
        bool_tensor = tf.constant([[True, False, True, False]], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        # Bits are packed as [1,0,1,0,0,0,0,0] (after padding to 8 bits)
        # Expected packed value is 160 (binary 10100000)
        expected_packed = tf.constant([[160]], dtype=tf.uint8)

        # Assert that the packed tensor matches the expected value
        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_base_case_uint16(self):
        """
        Test the base case with a simple boolean tensor and dtype tf.uint16
        """
        bool_tensor = tf.constant([[True, False, True, False]], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint16)

        # The packed bits are [1, 0, 1, 0, ... 0], starting from MSB (bit 15)
        # Expected packed value is 40960 (binary 1010000000000000)
        expected_packed = tf.constant([[40960]], dtype=tf.uint16)

        # Assert that the packed tensor matches the expected value
        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_all_false(self):
        """
        Test with a tensor of all False values
        """
        bool_tensor = tf.constant([[False] * 10], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        # After padding, bits are all zeros
        expected_packed = tf.constant([[0, 0]], dtype=tf.uint8)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_all_true(self):
        """
        Test with a tensor of all True values
        """
        bool_tensor = tf.constant([[True] * 10], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        # For tf.uint8, n_bits = 8
        # Bits after padding: [True]*10 + [False]*6 to make 16 bits
        # First packed integer corresponds to bits 15 to 8
        # Second packed integer corresponds to bits 7 to 0

        # Since we only have 10 bits, the bits will be:
        # Packed bits:
        #   First integer (bits 15-8): bits 0-7 (first 8 bits): True
        #   Second integer (bits 7-0): bits 8-9 (next 2 bits): True, True, then padding with zeros

        # For tf.uint8, the first packed integer is:
        # First packed integer: bits [True] * 8 => 255
        # Second packed integer: bits [True, True, False, False, False, False, False, False] => 192

        expected_packed = tf.constant([[255, 192]], dtype=tf.uint8)
        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_multiple_rows(self):
        """
        Test with multiple rows in the tensor
        """
        bool_tensor = tf.constant([
            [True, False, True, False],
            [False, True, False, True]
        ], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        # First row: 10100000 => 160
        # Second row: 01010000 => 80
        expected_packed = tf.constant([
            [160],
            [80]
        ], dtype=tf.uint8)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_non_multiple_of_n_bits(self):
        """
        Test with number of bits not a multiple of n_bits
        """
        bool_tensor = tf.constant([[True, False, True, False, True, False, True]], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        # Bits after padding: [1,0,1,0,1,0,1,0] => 170
        expected_packed = tf.constant([[170]], dtype=tf.uint8)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_large_tensor(self):
        """
        Test with a larger tensor of shape (4,16)
        """
        bool_tensor = tf.constant([[bool((i + j) % 2) for j in range(16)] for i in range(4)], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint16)

        # Expected values alternate between 0x5555 and 0xAAAA
        expected_values = []
        for i in range(4):
            val = 0xAAAA if i % 2 else 0x5555
            expected_values.append([val])
        expected_packed = tf.constant(expected_values, dtype=tf.uint16)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_different_dtypes(self):
        """
        Test the function with different integer data types
        """
        bool_tensor = tf.constant([[True] * 8], dtype=tf.bool)
        # Expected packed values for each dtype, considering MSB-first ordering
        expected_values = {
            tf.uint8: 255,
            tf.uint16: 65280,
            tf.uint32: 4278190080,
            tf.uint64: 18374686479671623680
        }
        for dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=dtype)
            expected_packed = tf.constant([[expected_values[dtype]]], dtype=dtype)
            tf.debugging.assert_equal(packed_tensor, expected_packed)


    def test_empty_tensor(self):
        """
        Test the function with an empty tensor
        """
        bool_tensor = tf.constant([[]], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)
        expected_packed = tf.constant([[]], dtype=tf.uint8)
        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_invalid_dtype(self):
        """
        Test passing a non-integer dtype to the function
        """
        bool_tensor = tf.constant([[True, False]], dtype=tf.bool)
        with self.assertRaises(ValueError):
            DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.float32)

    def test_non_boolean_input(self):
        """
        Test passing a non-boolean tensor as input
        """
        int_tensor = tf.constant([[1, 0, 1, 0]], dtype=tf.int32)
        with self.assertRaises(ValueError):
            DEPRECATED_pack_tensor_bits(int_tensor, dtype=tf.uint8)

    def test_rank_greater_than_two(self):
        """
        Test with an input tensor of rank greater than 2
        """
        bool_tensor = tf.constant([[[True, False], [False, True]]], dtype=tf.bool)
        with self.assertRaises(ValueError):
            DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

    def test_zero_dimensional_tensor(self):
        """
        Test the function with a zero-dimensional tensor
        """
        bool_tensor = tf.constant(True, dtype=tf.bool)
        with self.assertRaises(ValueError):
            DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

    def test_shape_mismatch(self):
        """
        Test when the input tensor shape is inappropriate (e.g., 1D tensor)
        """
        bool_tensor = tf.constant([True, False, True], dtype=tf.bool)
        with self.assertRaises(ValueError):
            DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

    def test_large_dtype(self):
        """
        Test with a large dtype like tf.uint64 to check correct bit packing
        """
        bool_tensor = tf.constant([[True] * 60], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint64)

        # Expected value: (2^60 - 1) shifted left by 4 bits due to padding
        expected_value = ((1 << 60) - 1) << 4
        expected_packed = tf.constant([[expected_value]], dtype=tf.uint64)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_packing_multiple_columns(self):
        """
        Test packing when there are more bits than the integer size (e.g., 16 bits into uint8)
        """
        bool_tensor = tf.constant([
            [True] * 9 + [False] * 7
        ], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        expected_packed = tf.constant([[255, 128]], dtype=tf.uint8)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_large_number_of_rows(self):
        """
        Test the function with a large number of rows
        """
        num_rows = 1000
        bool_tensor = tf.constant([[True] * 10] * num_rows, dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint16)

        expected_value = ((1 << 10) - 1) << 6
        expected_packed = tf.constant([[expected_value]] * num_rows, dtype=tf.uint16)

        tf.debugging.assert_equal(packed_tensor, expected_packed)

    def test_non_divisible_by_n_bits(self):
        """
        Another test case where the number of bits isn't divisible by n_bits
        """
        bool_tensor = tf.constant([
            [True, False, True, False, True]  # 5 bits
        ], dtype=tf.bool)
        packed_tensor = DEPRECATED_pack_tensor_bits(bool_tensor, dtype=tf.uint8)

        expected_packed = tf.constant([[168]], dtype=tf.uint8)  # 0b10101000

        tf.debugging.assert_equal(packed_tensor, expected_packed)
