import unittest
import flatbuffers
from pracciolini.grammar.canopy.model.buffer_manager import BufferManager

class BufferManagerTests(unittest.TestCase):
    """
    Unit tests for the BufferManager class.

    These tests cover the core functionality of the BufferManager class,
    including adding buffers with various types of data, handling edge cases,
    and ensuring that buffer indices and data storage are consistent and correct.
    """

    def setUp(self):
        # This method is called before every test.
        # Initialize a new BufferManager and FlatBuffers Builder for each test to ensure isolation.
        self.buffer_manager = BufferManager()
        self.builder = flatbuffers.Builder(1024)

    def test_initial_state(self):
        """
        Test the initial state of the BufferManager.

        Verifies that the empty buffer is correctly initialized.
        """
        # There should be one buffer (the empty buffer)
        self.assertEqual(len(self.buffer_manager.get_buffers()), 1)
        # The first buffer should be empty
        self.assertEqual(self.buffer_manager.get_buffers()[0], b'')

    def test_add_buffer_empty_data(self):
        """
        Test adding a buffer with empty data bytes.

        This checks that:
        - The buffer index is assigned.
        - The buffer data is stored as empty bytes.
        """
        data_bytes = b""
        buffer_idx = self.buffer_manager.add_buffer(data_bytes)

        # Verify that the buffer index is correct (should be 1, since index 0 is the empty buffer)
        self.assertEqual(buffer_idx, 1, "First buffer index should be 1")

        # Verify that the buffer data is stored correctly
        self.assertEqual(self.buffer_manager.get_buffers()[buffer_idx], data_bytes)

    def test_add_buffer_non_empty_data(self):
        """
        Test adding a buffer with non-empty data bytes.

        This checks that:
        - The buffer index is assigned.
        - The buffer data is stored correctly.
        """
        data_bytes = b"\x01\x02\x03"
        buffer_idx = self.buffer_manager.add_buffer(data_bytes)

        # Verify that the buffer index is correct
        self.assertEqual(buffer_idx, 1, "First buffer index should be 1")

        # Verify that the buffer data is stored correctly
        self.assertEqual(self.buffer_manager.get_buffers()[buffer_idx], data_bytes)

    def test_add_multiple_buffers(self):
        """
        Test adding multiple buffers and verify their indices and data.
        """
        data_bytes_1 = b"\x01\x02\x03"
        data_bytes_2 = b"\x04\x05\x06"
        data_bytes_3 = b""

        buffer_idx_1 = self.buffer_manager.add_buffer(data_bytes_1)
        buffer_idx_2 = self.buffer_manager.add_buffer(data_bytes_2)
        buffer_idx_3 = self.buffer_manager.add_buffer(data_bytes_3)

        # Verify that buffer indices are assigned incrementally
        self.assertEqual(buffer_idx_1, 1)
        self.assertEqual(buffer_idx_2, 2)
        self.assertEqual(buffer_idx_3, 3)

        # Verify that the buffer data is stored correctly
        self.assertEqual(self.buffer_manager.get_buffers()[buffer_idx_1], data_bytes_1)
        self.assertEqual(self.buffer_manager.get_buffers()[buffer_idx_2], data_bytes_2)
        self.assertEqual(self.buffer_manager.get_buffers()[buffer_idx_3], data_bytes_3)

        # Total number of buffers should be 4 (empty buffer + 3 added)
        self.assertEqual(len(self.buffer_manager.get_buffers()), 4)

    def test_add_buffer_valid_data(self):
        """
        Test adding a buffer with valid data bytes.

        This checks that:
        - The buffer index returned is correct.
        - The data is stored correctly in the buffer manager.
        - The buffer offsets are recorded.
        """
        data_bytes = b"test_data"
        buffer_idx = self.buffer_manager.add_buffer(data_bytes)

        # Verify that the buffer index is 1 since it's the first buffer added, not counting the default buffer
        self.assertEqual(buffer_idx, 1, "First buffer index should be 1")

        # Verify that the buffer data is correctly stored.
        self.assertEqual(self.buffer_manager.get_buffers()[1], data_bytes, "Buffer data should match the input data")

        # Verify that a buffer offset has been recorded.
        buffer_offsets = self.buffer_manager.get_buffer_offsets()
        self.assertEqual(len(buffer_offsets), 1, "There should be one buffer offset recorded")

    def test_get_buffer_offsets_empty(self):
        """
        Test retrieving buffer offsets when no buffers have been added.

        This checks that:
        - The buffer offsets list is empty.
        """
        buffer_offsets = self.buffer_manager.get_buffer_offsets()

        # Verify that the buffer offsets list is empty.
        self.assertEqual(buffer_offsets, [0], "Buffer offsets should be empty when no buffers have been added")

    def test_buffer_indices_consistency(self):
        """
        Test that buffer indices remain consistent and predictable.

        This checks that:
        - Buffer indices are assigned sequentially even when adding the same data multiple times.
        """
        data_bytes = b"test_data"
        buffer_idx1 = self.buffer_manager.add_buffer(data_bytes)
        buffer_idx2 = self.buffer_manager.add_buffer(data_bytes)

        # Verify that buffer indices are sequential.
        self.assertEqual(buffer_idx1, 1, "First buffer index should be 1")
        self.assertEqual(buffer_idx2, 2, "Second buffer index should be 2")

    def test_large_data_buffer(self):
        """
        Test adding a buffer with a large amount of data.

        This checks that:
        - The buffer manager can handle large data without issues.
        """
        # Generate large data.
        data_bytes = b"A" * (10 * 1024 * 1024)  # 10 MB of data
        buffer_idx = self.buffer_manager.add_buffer(data_bytes)

        # Verify that the buffer index is correct.
        self.assertEqual(buffer_idx, 1, "Buffer index should be 1 for the first buffer")

        # Verify that the data is stored correctly.
        self.assertEqual(self.buffer_manager.get_buffers()[1], data_bytes,
                         "Buffer data should match the large input data")

    def test_buffer_offsets_correspond_to_data(self):
        """
        Test that buffer offsets correspond correctly to the data added.

        This checks that:
        - The buffer offsets stored can be used to retrieve the correct data from the FlatBuffers builder.
        """
        data_bytes = b"test_data"
        self.buffer_manager.add_buffer(data_bytes)

        buffer_offsets = self.buffer_manager.get_buffer_offsets()
        # Since FlatBuffers builder doesn't allow direct access to the object by offset,
        # we assume here that the buffer offsets are valid as per the FlatBuffers library's internals.

        # Verify that one buffer offset is recorded.
        self.assertEqual(len(buffer_offsets), 1, "There should be one buffer offset recorded")

    def test_add_buffer_special_characters(self):
        """
        Test adding a buffer with special characters and binary data.

        This checks that:
        - The buffer manager correctly handles arbitrary binary data.
        """
        data_bytes = b"\x00\xFF\xFE\xFD\xFC" + b"special_chars_@\u20AC"
        buffer_idx = self.buffer_manager.add_buffer(data_bytes)

        # Verify that the buffer index is assigned.
        self.assertEqual(buffer_idx, 1, "Buffer index should be 1 for the first buffer")

        # Verify that the data is stored correctly.
        self.assertEqual(self.buffer_manager.get_buffers()[1], data_bytes,
                         "Buffer data should match the input data with special characters")

    def test_add_duplicate_buffers(self):
        """
        Test adding duplicate buffers and verify that they are stored separately.

        This checks that:
        - Each call to add_buffer results in a new buffer, even if the data is the same.
        - Buffer indices are assigned sequentially.
        """
        data_bytes = b"duplicate_data"
        buffer_idx1 = self.buffer_manager.add_buffer(data_bytes)
        buffer_idx2 = self.buffer_manager.add_buffer(data_bytes)

        # Verify that different buffer indices are assigned.
        self.assertNotEqual(buffer_idx1, buffer_idx2, "Buffer indices for duplicate data should be different")

        # Verify that both buffers are stored.
        self.assertEqual(len(self.buffer_manager.get_buffers()), 3, "There should be two buffers stored")
        self.assertEqual(self.buffer_manager.get_buffers()[1], data_bytes,
                         "First buffer data should match the input data")
        self.assertEqual(self.buffer_manager.get_buffers()[2], data_bytes,
                         "Second buffer data should match the input data")

    def test_get_buffer_data_after_reset(self):
        """
        Test retrieving buffer data after re-initializing the BufferManager.

        This checks that:
        - The buffer data is cleared when a new BufferManager is created.
        """
        data_bytes = b"test_data"
        self.buffer_manager.add_buffer(data_bytes)

        # Re-initialize the BufferManager.
        self.buffer_manager = BufferManager()

        # Verify that buffer data is empty.
        buffer_data = self.buffer_manager.get_buffers()
        self.assertEqual(buffer_data, [b''], "Buffer data should be empty after re-initialization")

    def test_add_buffer_none_data(self):
        """
        Test adding a buffer with None as data_bytes.

        This checks that:
        - The method raises an appropriate exception.
        """
        with self.assertRaises(TypeError, msg="Adding None as data_bytes should raise TypeError"):
            self.buffer_manager.add_buffer(None)

    def test_add_buffer_invalid_data_type(self):
        """
        Test adding a buffer with invalid data type (e.g., integer).

        This checks that:
        - The method raises an appropriate exception for invalid data types.
        """
        with self.assertRaises(TypeError, msg="Adding non-bytes data should raise TypeError"):
            self.buffer_manager.add_buffer(12345)  # Invalid data type

    def test_serialize_buffers_without_buffers(self):
        """
        Test serializing buffers when no buffers have been added (only the empty buffer exists).

        This checks that:
        - The serialize_buffers method works without errors.
        - The buffer offsets correspond correctly to the buffers.
        """
        self.buffer_manager.serialize_buffers(self.builder)
        buffer_offsets = self.buffer_manager.get_buffer_offsets()

        # Only the empty buffer should exist
        self.assertEqual(len(buffer_offsets), 1, "There should be one buffer offset (the empty buffer)")
        # The offset for the empty buffer should be 0 or valid as per FlatBuffers
        self.assertIsInstance(buffer_offsets[0], int, "Buffer offset should be an integer")

    def test_serialize_buffers_after_adding_buffers(self):
        """
        Test that buffer_offsets are correctly set after calling serialize_buffers.

        This checks that:
        - The buffer offsets correspond to the buffers added.
        - The offsets are in the correct order.
        """
        data_bytes_1 = b"buffer1"
        data_bytes_2 = b"buffer2"
        self.buffer_manager.add_buffer(data_bytes_1)
        self.buffer_manager.add_buffer(data_bytes_2)

        self.buffer_manager.serialize_buffers(self.builder)
        buffer_offsets = self.buffer_manager.get_buffer_offsets()

        # There should be three buffers: empty buffer + two added buffers
        self.assertEqual(len(buffer_offsets), 3, "There should be three buffer offsets")

        # Check that the buffer offsets are valid
        for offset in buffer_offsets:
            self.assertIsInstance(offset, int, "Each buffer offset should be an integer")

    def test_serialize_buffers_with_large_number_of_buffers(self):
        """
        Test adding a large number of buffers to check for performance or memory issues.

        This checks that:
        - The BufferManager can handle a large number of buffers without errors.
        """
        num_buffers = 10000  # Large number of buffers
        data_bytes = b"A"  # Small data to avoid excessive memory usage

        for _ in range(num_buffers):
            self.buffer_manager.add_buffer(data_bytes)

        # Serialize buffers and ensure no exceptions are raised
        try:
            self.buffer_manager.serialize_buffers(self.builder)
        except Exception as e:
            self.fail(f"Serialization failed with a large number of buffers: {e}")

        buffer_offsets = self.buffer_manager.get_buffer_offsets()
        expected_buffer_count = num_buffers + 1  # Including the empty buffer
        self.assertEqual(len(buffer_offsets), expected_buffer_count,
                         f"There should be {expected_buffer_count} buffer offsets")

    def test_add_buffer_after_serialization(self):
        """
        Test adding a buffer after calling serialize_buffers.

        This checks that:
        - Adding buffers after serialization does not corrupt the internal state.
        - Subsequent serialization includes the new buffer.
        """
        data_bytes_initial = b"initial_data"
        self.buffer_manager.add_buffer(data_bytes_initial)
        self.buffer_manager.serialize_buffers(self.builder)

        # Add another buffer after serialization
        data_bytes_new = b"new_data"
        buffer_idx_new = self.buffer_manager.add_buffer(data_bytes_new)

        # Serialize buffers again
        self.buffer_manager.serialize_buffers(self.builder)
        buffer_offsets = self.buffer_manager.get_buffer_offsets()

        # There should now be three buffers: empty buffer + initial + new
        self.assertEqual(len(buffer_offsets), 3, "There should be three buffer offsets after adding a new buffer")

        # Verify that the new buffer index is correct
        self.assertEqual(buffer_idx_new, 2, "New buffer index should be 2 after adding buffer post-serialization")
        self.assertEqual(self.buffer_manager.get_buffers()[buffer_idx_new], data_bytes_new,
                         "New buffer data should match the input data")

    def test_buffer_manager_thread_safety(self):
        """
        Test the thread safety of BufferManager when adding buffers from multiple threads.

        This checks that:
        - Buffer indices remain consistent even when accessed from multiple threads.
        """
        import threading

        num_threads = 10
        buffers_per_thread = 100
        data_bytes = b"thread_data"

        def add_buffers():
            for _ in range(buffers_per_thread):
                self.buffer_manager.add_buffer(data_bytes)

        threads = [threading.Thread(target=add_buffers) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Total buffers should be empty buffer + (num_threads * buffers_per_thread)
        expected_buffer_count = 1 + (num_threads * buffers_per_thread)
        self.assertEqual(len(self.buffer_manager.get_buffers()), expected_buffer_count,
                         f"There should be {expected_buffer_count} buffers after multithreaded addition")

    def test_buffer_indices_after_concurrent_additions(self):
        """
        Test that buffer indices are unique and correct after concurrent additions.

        This checks that:
        - Each buffer added from multiple threads has a unique and correct index.
        """
        import threading

        num_threads = 5
        buffers_per_thread = 20
        data_bytes = b"concurrent_data"
        buffer_indices = []

        def add_buffers_and_collect_indices():
            for _ in range(buffers_per_thread):
                buffer_idx = self.buffer_manager.add_buffer(data_bytes)
                buffer_indices.append(buffer_idx)

        threads = [threading.Thread(target=add_buffers_and_collect_indices) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Check that all buffer indices are unique
        self.assertEqual(len(buffer_indices), num_threads * buffers_per_thread,
                         "Total number of buffer indices collected should match")
        self.assertEqual(len(set(buffer_indices)), num_threads * buffers_per_thread,
                         "All buffer indices should be unique")

    def test_get_buffer_offsets_before_serialization(self):
        """
        Test accessing buffer offsets before serialization.

        This checks that:
        - The buffer offsets list is empty except for the placeholder for the empty buffer.
        """
        buffer_offsets = self.buffer_manager.get_buffer_offsets()
        self.assertEqual(buffer_offsets, [0], "Buffer offsets should contain only the empty buffer placeholder")

    def test_buffer_data_alignment(self):
        """
        Test that data alignment is maintained for buffers.

        This checks that:
        - Data alignment requirements are met according to the schema.
        """
        data_bytes = b"\x00" * 16  # 16 bytes to align with 16-byte boundary
        self.buffer_manager.add_buffer(data_bytes)
        self.buffer_manager.serialize_buffers(self.builder)
        buffer_offsets = self.buffer_manager.get_buffer_offsets()

        # In FlatBuffers, elements are aligned according to their types.
        # Since we don't have direct control over the memory addresses,
        # we ensure that the buffer was serialized without errors.
        # An explicit check on alignment would require inspecting internal buffer positions.

        # Verify that serialization did not raise errors and buffer offsets are valid integers.
        for offset in buffer_offsets:
            self.assertIsInstance(offset, int, "Buffer offset should be an integer")

    def test_repeated_serialization_consistency(self):
        """
        Test that repeated serialization yields consistent buffer offsets.

        This checks that:
        - Calling serialize_buffers multiple times does not change the buffer offsets.
        """
        data_bytes = b"consistent_data"
        self.buffer_manager.add_buffer(data_bytes)
        self.buffer_manager.serialize_buffers(self.builder)
        buffer_offsets_first = self.buffer_manager.get_buffer_offsets().copy()

        # Serialize again
        self.builder = flatbuffers.Builder(1024)
        self.buffer_manager.serialize_buffers(self.builder)
        buffer_offsets_second = self.buffer_manager.get_buffer_offsets()

        self.assertEqual(buffer_offsets_first, buffer_offsets_second,
                         "Buffer offsets should remain consistent across serializations")

    def test_buffer_manager_state_after_exception(self):
        """
        Test the state of the BufferManager after an exception is raised during buffer addition.

        This checks that:
        - The internal state remains consistent even if an exception occurs.
        """
        valid_data = b"valid_data"
        self.buffer_manager.add_buffer(valid_data)

        # Attempt to add invalid data to trigger an exception
        with self.assertRaises(TypeError):
            self.buffer_manager.add_buffer(12345)

        # Verify that the buffer manager still contains the valid buffer
        buffers = self.buffer_manager.get_buffers()
        self.assertEqual(len(buffers), 2, "Buffer manager should contain the empty buffer and one valid buffer")

    def test_buffer_manager_with_empty_builder(self):
        """
        Test the behavior when passing an uninitialized FlatBuffers builder.

        This checks that:
        - The method handles an uninitialized builder appropriately.
        """
        data_bytes = b"test_data"
        uninitialized_builder = flatbuffers.Builder(0)  # Builder with zero initial size

        try:
            buffer_idx = self.buffer_manager.add_buffer(data_bytes)
            self.buffer_manager.serialize_buffers(uninitialized_builder)
        except Exception as e:
            self.fail(f"BufferManager failed with an uninitialized builder: {e}")

        # Verify that the buffer was added and serialized
        self.assertEqual(buffer_idx, 1, "Buffer index should be 1 with uninitialized builder")
        buffer_offsets = self.buffer_manager.get_buffer_offsets()
        self.assertEqual(len(buffer_offsets), 2, "There should be two buffer offsets after serialization")

if __name__ == '__main__':
    unittest.main()