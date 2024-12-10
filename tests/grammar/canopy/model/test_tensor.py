import unittest
import tensorflow as tf
import flatbuffers
import numpy as np
import threading

# Assuming the BufferManager and Tensor classes are available
from pracciolini.grammar.canopy.model.buffer_manager import BufferManager
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.io import Tensor as IoTensor


class TensorCoreTests(unittest.TestCase):

    def setUp(self):
        # Set up common resources for the tests
        self.builder = flatbuffers.Builder(1024)
        self.buffer_manager = BufferManager()

    def test_tensor_initialization_valid(self):
        # Test initializing Tensor with a valid tf.Tensor
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='test_tensor')
        self.assertEqual(tensor.tf_tensor.numpy().tolist(), [1, 2, 3])
        self.assertEqual(tensor.name, 'test_tensor')

    def test_tensor_initialization_invalid(self):
        # Test initializing Tensor with invalid input
        with self.assertRaises(TypeError):
            Tensor([1, 2, 3], name='invalid_tensor')  # Not a tf.Tensor

    def test_tensor_serialization(self):
        # Test serializing a Tensor
        tf_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='float_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # Deserialize to check correctness
        tensor_data = IoTensor.Tensor.GetRootAs(buf, 0)
        shape = [tensor_data.Shape(i) for i in range(tensor_data.ShapeLength())]
        self.assertEqual(shape, [2, 2])
        self.assertEqual(tensor_data.Type(), 160)  # FLOAT32 enum value
        self.assertEqual(tensor_data.Name().decode('utf-8'), 'float_tensor')
        self.assertEqual(tensor_data.BufferIdx(), 1)

        # Check buffer data
        buffers = self.buffer_manager.get_buffers()
        self.assertEqual(len(buffers), 2)
        expected_data = tf_tensor.numpy().tobytes()
        self.assertEqual(buffers[1], expected_data)

    def test_tensor_deserialization(self):
        # Build an IoTensor manually
        tensor_data = tf.constant([5, 6, 7, 8], dtype=tf.uint16).numpy()
        buffer_data = tensor_data.tobytes()
        self.buffer_manager.add_buffer(buffer_data)

        # Create the FlatBuffer for the Tensor
        shape = [4]
        IoTensor.TensorStartShapeVector(self.builder, len(shape))
        for dim in reversed(shape):
            self.builder.PrependInt32(dim)
        shape_vector = self.builder.EndVector(len(shape))

        name_offset = self.builder.CreateString('uint16_tensor')
        IoTensor.TensorStart(self.builder)
        IoTensor.TensorAddShape(self.builder, shape_vector)
        IoTensor.TensorAddType(self.builder, 16)  # UINT16 enum value
        IoTensor.TensorAddBufferIdx(self.builder, 1)  # Updated buffer index
        IoTensor.TensorAddName(self.builder, name_offset)
        tensor_offset = IoTensor.TensorEnd(self.builder)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # Deserialize using from_graph
        io_tensor = IoTensor.Tensor.GetRootAs(buf, 0)
        buffers = self.buffer_manager.get_buffers()
        tensor = Tensor.from_graph(io_tensor, buffers)
        self.assertEqual(tensor.name.decode('utf-8'), 'uint16_tensor')
        self.assertEqual(tensor.tf_tensor.numpy().tolist(), [5, 6, 7, 8])
        self.assertEqual(tensor.tf_tensor.dtype, tf.uint16)

    def test_tensor_dynamic_shape(self):
        # Test tensor with a shape that includes None (dynamic dimension)
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        tf_tensor = tf.constant(data)
        # Define the shape with dynamic first dimension
        dynamic_shape = [None, 3]
        tensor = Tensor(tf_tensor, name='dynamic_tensor', shape=dynamic_shape)

        # Serialize the tensor
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # Deserialize to check correctness
        io_tensor = IoTensor.Tensor.GetRootAs(buf, 0)
        shape = [io_tensor.Shape(i) for i in range(io_tensor.ShapeLength())]
        self.assertEqual(shape, [-1, 3])  # Expect -1 for dynamic dimension
        buffers = self.buffer_manager.get_buffers()
        self.assertEqual(len(buffers), 2)  # Expect 2 buffers: empty buffer + data buffer
        self.assertEqual(buffers[1], data.tobytes())

        # Now test deserialization
        tensor_deserialized = Tensor.from_graph(io_tensor, buffers)
        self.assertEqual(tensor_deserialized.name.decode('utf-8'), 'dynamic_tensor')
        # The tensor shape in TensorFlow should be [2, 3] after data is assigned
        self.assertEqual(tensor_deserialized.tf_tensor.shape.as_list(), [2, 3])
        self.assertTrue(np.array_equal(tensor_deserialized.tf_tensor.numpy(), data))

    def test_tensor_unsupported_dtype(self):
        # Test initializing Tensor with unsupported dtype
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
        with self.assertRaises(ValueError):
            tensor = Tensor(tf_tensor, name='int32_tensor')
            tensor.to_graph(self.builder, self.buffer_manager)

    def test_tensor_empty(self):
        # Test Tensor with empty data
        tf_tensor = tf.constant([], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='empty_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # Deserialize to check correctness
        io_tensor = IoTensor.Tensor.GetRootAs(buf, 0)
        shape = [io_tensor.Shape(i) for i in range(io_tensor.ShapeLength())]
        self.assertEqual(shape, [0])
        buffers = self.buffer_manager.get_buffers()
        self.assertEqual(len(buffers), 2)  # Expect 2 buffers: empty buffer + data buffer
        self.assertEqual(buffers[0], b'')  # Empty bytes

    def test_tensor_large_dimensions(self):
        # Test Tensor with large dimensions
        tf_tensor = tf.zeros((1024, 1024), dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='large_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        # We won't deserialize here due to size, but ensure no exceptions were raised
        self.assertTrue(tensor_offset > 0)

    def test_tensor_with_nan_and_inf(self):
        # Test Tensor containing NaN and Inf values
        tf_tensor = tf.constant([float('nan'), float('inf'), -float('inf')], dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='nan_inf_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()
        # Deserialize and check
        IoTensor.Tensor.GetRootAs(buf, 0)
        buffers = self.buffer_manager.get_buffers()
        data_bytes = buffers[1]
        data = np.frombuffer(data_bytes, dtype=np.float32)
        self.assertTrue(np.isnan(data[0]))
        self.assertTrue(np.isinf(data[1]) and data[1] > 0)
        self.assertTrue(np.isinf(data[2]) and data[2] < 0)

    def test_tensor_zero_dimensions(self):
        # Test Tensor with zero dimensions (scalar)
        tf_tensor = tf.constant(42, dtype=tf.uint32)
        tensor = Tensor(tf_tensor, name='scalar_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # Deserialize and check
        io_tensor = IoTensor.Tensor.GetRootAs(buf, 0)
        shape = [io_tensor.Shape(i) for i in range(io_tensor.ShapeLength())]
        self.assertEqual(shape, [])  # Scalar has empty shape
        buffers = self.buffer_manager.get_buffers()
        data_bytes = buffers[1]
        data = np.frombuffer(data_bytes, dtype=np.uint32)
        self.assertEqual(data.tolist(), [42])

    def test_tensor_helper_methods(self):
        # Testing internal helper methods (if applicable)
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='test_helpers')

        # Test _get_tensor_data_bytes
        data_bytes = tensor._get_tensor_data_bytes()
        expected_bytes = tf_tensor.numpy().tobytes()
        self.assertEqual(data_bytes, expected_bytes)

        # Test _map_tf_dtype_to_tensortype
        tensortype = tensor._map_tf_dtype_to_tensortype(tf.uint8)
        self.assertEqual(tensortype, 8)  # UINT8 enum value

        # Test _map_tensortype_to_tf_dtype
        dtype = tensor._map_tensortype_to_tf_dtype(8)
        self.assertEqual(dtype, tf.uint8)

        # Test with unsupported dtype
        with self.assertRaises(ValueError):
            tensor._map_tf_dtype_to_tensortype(tf.int32)

    def test_tensor_none_name(self):
        # Test initializing Tensor without a name
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor)
        self.assertIsNone(tensor.name)
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # Deserialize and check that Name is None
        io_tensor = IoTensor.Tensor.GetRootAs(buf, 0)
        self.assertIsNone(io_tensor.Name())

    def test_tensor_negative_buffer_idx(self):
        # Test behavior when buffer_idx is out of range during deserialization
        # Build an IoTensor manually with invalid buffer_idx
        shape = [2, 2]
        IoTensor.TensorStartShapeVector(self.builder, len(shape))
        for dim in reversed(shape):
            self.builder.PrependInt32(dim)
        shape_vector = self.builder.EndVector(len(shape))

        name_offset = self.builder.CreateString('invalid_buffer_idx')
        IoTensor.TensorStart(self.builder)
        IoTensor.TensorAddShape(self.builder, shape_vector)
        IoTensor.TensorAddType(self.builder, 160)  # FLOAT32
        IoTensor.TensorAddBufferIdx(self.builder, 999)  # Invalid buffer index
        IoTensor.TensorAddName(self.builder, name_offset)
        tensor_offset = IoTensor.TensorEnd(self.builder)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        buf = self.builder.Output()

        # No buffers added to buffer manager
        buffers = []
        io_tensor = IoTensor.Tensor.GetRootAs(buf, 0)
        # Attempt to deserialize should raise IndexError
        with self.assertRaises(IndexError):
            Tensor.from_graph(io_tensor, buffers)

    def test_tensor_complex_dtype(self):
        # Test Tensor with unsupported complex dtype
        tf_tensor = tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)
        with self.assertRaises(ValueError):
            tensor = Tensor(tf_tensor, name='complex_tensor')
            tensor.to_graph(self.builder, self.buffer_manager)

    def test_constructor_with_tf_variable(self):
        # Test initializing Tensor with a tf.Variable instead of tf.Tensor
        tf_var = tf.Variable([10, 20, 30], dtype=tf.float32)
        tensor = Tensor(tf_var, name='variable_tensor')
        self.assertTrue(isinstance(tensor.tf_tensor, tf.Variable))
        self.assertEqual(tensor.tf_tensor.numpy().tolist(), [10, 20, 30])

    def test_constructor_with_special_character_name(self):
        # Test initializing Tensor with special characters in the name
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        special_name = "tensor@#%&*!"
        tensor = Tensor(tf_tensor, name=special_name)
        self.assertEqual(tensor.name, special_name)

    def test_constructor_with_sparse_tensor(self):
        # Test initializing Tensor with a tf.SparseTensor
        sparse_indices = [[0, 0], [1, 2]]
        sparse_values = [1, 2]
        sparse_shape = [3, 4]
        sparse_tensor = tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=sparse_shape)
        with self.assertRaises(TypeError):
            Tensor(sparse_tensor, name='sparse_tensor')

    def test_constructor_with_ragged_tensor(self):
        # Test initializing Tensor with a tf.RaggedTensor
        ragged_tensor = tf.ragged.constant([[1, 2], [3]])
        with self.assertRaises(TypeError):
            Tensor(ragged_tensor, name='ragged_tensor')

    def test_serialization_with_uninitialized_tensor(self):
        # Test serialization when tensor data is uninitialized
        # Note: In TensorFlow 2.x, all tensors are eagerly executed and initialized.
        # To simulate uninitialized data, we can create a tensor that depends on a variable
        v = tf.Variable(tf.ones([2, 2]), dtype=tf.float32)
        tf_tensor = v + 1  # This should be computed
        tensor = Tensor(tf_tensor, name='uninitialized_tensor')
        # Forcing variable initialization
        v.assign(tf.zeros([2, 2]))
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.assertTrue(tensor_offset > 0)

    def test_deserialization_with_corrupted_buffer(self):
        # Test deserialization when the buffer data is corrupted
        # Build a valid tensor first
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='corrupted_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.builder.Finish(tensor_offset)

        # Get the buffer index used in the serialized tensor
        io_tensor = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)
        buffer_idx = io_tensor.BufferIdx()

        # Prepare corrupted buffers
        # Since BufferManager adds an empty buffer at index 0, we need to replicate that
        # Create an empty buffer for index 0
        empty_buffer = b''
        # Place the corrupted data at the correct buffer index
        corrupted_data = b'\x00\x01'  # Invalid data (too short)
        # Construct buffers list with correct indices
        # Ensure the list has at least buffer_idx + 1 elements
        corrupted_buffers = [empty_buffer] * buffer_idx + [corrupted_data]

        # Attempt to deserialize with corrupted buffer
        with self.assertRaises(ValueError):
            Tensor.from_graph(io_tensor, corrupted_buffers)

    def test_serialization_thread_safety(self):
        # Test serializing tensors in parallel threads
        def serialize_tensor(tensor, builder, buffer_manager):
            tensor.to_graph(builder, buffer_manager)

        tf_tensor1 = tf.constant([1], dtype=tf.uint8)
        tf_tensor2 = tf.constant([2], dtype=tf.uint8)
        tensor1 = Tensor(tf_tensor1, name='tensor1')
        tensor2 = Tensor(tf_tensor2, name='tensor2')

        builder1 = flatbuffers.Builder(1024)
        builder2 = flatbuffers.Builder(1024)
        buffer_manager1 = BufferManager()
        buffer_manager2 = BufferManager()

        thread1 = threading.Thread(target=serialize_tensor, args=(tensor1, builder1, buffer_manager1))
        thread2 = threading.Thread(target=serialize_tensor, args=(tensor2, builder2, buffer_manager2))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # If no exceptions occurred, the test passes

    def test_deserialization_with_dtype_mismatch(self):
        # Test deserialization when tensor's dtype is mismatched with buffer content
        # Create a tensor and serialize it
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='dtype_mismatch_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.builder.Finish(tensor_offset)
        buffers = self.buffer_manager.get_buffers()

        # Manually modify the dtype in the serialized data
        io_tensor = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)

        # Let's create a new IoTensor with incorrect Type (e.g., FLOAT32 instead of UINT8)
        # First, create any strings or vectors needed
        name_str = io_tensor.Name()
        if name_str is not None:
            name_offset = self.builder.CreateString(name_str.decode())
        else:
            name_offset = None

        # Also, create the shape vector
        shape_length = io_tensor.ShapeLength()
        IoTensor.TensorStartShapeVector(self.builder, shape_length)
        for i in reversed(range(shape_length)):
            self.builder.PrependInt32(io_tensor.Shape(i))
        shape_vector = self.builder.EndVector(shape_length)

        # Now start the Tensor object
        IoTensor.TensorStart(self.builder)
        # Add the created fields
        IoTensor.TensorAddShape(self.builder, shape_vector)
        IoTensor.TensorAddType(self.builder, 160)  # FLOAT32 enum value
        IoTensor.TensorAddBufferIdx(self.builder, io_tensor.BufferIdx())
        if name_offset is not None:
            IoTensor.TensorAddName(self.builder, name_offset)
        new_tensor_offset = IoTensor.TensorEnd(self.builder)
        self.builder.Finish(new_tensor_offset)

        io_tensor_mismatch = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)

        # Verify that the Type is indeed set to FLOAT32
        self.assertEqual(io_tensor_mismatch.Type(), 160)

        # Attempt to deserialize
        with self.assertRaises(ValueError):
            Tensor.from_graph(io_tensor_mismatch, buffers)

    def test_high_dimensional_tensor(self):
        # Test serialization/deserialization of high-dimensional tensors
        shape = [2] * 8  # Shape: [2, 2, 2, 2, 2, 2, 2, 2]
        tf_tensor = tf.zeros(shape, dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='high_dim_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.assertTrue(tensor_offset > 0)
        self.builder.Finish(tensor_offset)
        io_tensor = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)

        # Deserialize and verify
        buffers = self.buffer_manager.get_buffers()
        tensor_deserialized = Tensor.from_graph(io_tensor, buffers)
        self.assertEqual(tensor_deserialized.tf_tensor.shape.as_list(), shape)

    def test_special_numeric_values(self):
        # Test tensor containing subnormal (denormal) float values
        subnormal_value = np.nextafter(0, 1, dtype=np.float32)
        tf_tensor = tf.constant([subnormal_value], dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='subnormal_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)

        # Finish the buffer
        self.builder.Finish(tensor_offset)
        io_tensor = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)
        buffers = self.buffer_manager.get_buffers()
        # Deserialize and check
        tensor_deserialized = Tensor.from_graph(io_tensor, buffers)
        deserialized_value = tensor_deserialized.tf_tensor.numpy()[0]
        self.assertEqual(deserialized_value, subnormal_value)

    def test_serialization_of_large_tensor(self):
        # Test serialization of a very large tensor
        large_size = 1000000  # One million elements
        tf_tensor = tf.zeros([large_size], dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='large_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.assertTrue(tensor_offset > 0)
        # We won't attempt to deserialize due to size constraints

    def test_tensor_with_string_data(self):
        # Test tensor containing string data, which is unsupported
        tf_tensor = tf.constant(['a', 'b', 'c'], dtype=tf.string)
        with self.assertRaises(ValueError):
            tensor = Tensor(tf_tensor, name='string_tensor')
            tensor.to_graph(self.builder, self.buffer_manager)

    def test_tensor_with_boolean_data(self):
        # Test tensor containing boolean data
        tf_tensor = tf.constant([True, False, True], dtype=tf.bool)
        with self.assertRaises(ValueError):
            tensor = Tensor(tf_tensor, name='bool_tensor')
            tensor.to_graph(self.builder, self.buffer_manager)

    def test_tensor_with_custom_dtype(self):
        # Test tensor with a custom or unknown dtype
        custom_dtype = tf.dtypes.as_dtype('bfloat16')
        tf_tensor = tf.constant([1, 2, 3], dtype=custom_dtype)
        with self.assertRaises(ValueError):
            tensor = Tensor(tf_tensor, name='custom_dtype_tensor')
            tensor.to_graph(self.builder, self.buffer_manager)

    def test_tensor_with_incorrect_shape_in_deserialization(self):
        # Test deserialization where buffer data does not match the shape
        # Create a tensor and serialize it
        tf_tensor = tf.constant([1, 2, 3, 4], dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='mismatch_shape_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.builder.Finish(tensor_offset)
        buffers = self.buffer_manager.get_buffers()

        # Modify the shape in the serialized data
        io_tensor = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)

        # Manually create a shape that doesn't match the data
        incorrect_shape_array = np.array([2, 3], dtype=np.int32)
        incorrect_shape_vector = self.builder.CreateNumpyVector(incorrect_shape_array)

        # Create the name string before starting the object
        name_str = io_tensor.Name().decode() if io_tensor.Name() else ''
        name_offset = self.builder.CreateString(name_str)

        # Now start the Tensor object
        IoTensor.TensorStart(self.builder)
        IoTensor.TensorAddShape(self.builder, incorrect_shape_vector)
        IoTensor.TensorAddType(self.builder, io_tensor.Type())
        IoTensor.TensorAddBufferIdx(self.builder, io_tensor.BufferIdx())
        IoTensor.TensorAddName(self.builder, name_offset)
        new_tensor_offset = IoTensor.TensorEnd(self.builder)
        self.builder.Finish(new_tensor_offset)

        # Deserialize with incorrect shape
        io_tensor_mismatch_shape = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)

        # Attempt to deserialize
        with self.assertRaises(ValueError):
            Tensor.from_graph(io_tensor_mismatch_shape, buffers)

    def test_tensor_serialization_with_empty_buffer_manager(self):
        # Test serialization when the buffer manager is not properly initialized
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='empty_buffer_manager')
        buffer_manager = None  # Not initialized
        with self.assertRaises(AttributeError):
            tensor.to_graph(self.builder, buffer_manager)

    def test_tensor_serialization_with_large_flatbuffer_builder(self):
        # Test serialization with a very large FlatBuffers builder
        large_builder = flatbuffers.Builder(1024 * 1024 * 50)  # 50 MB builder
        tf_tensor = tf.constant([1], dtype=tf.float32)
        tensor = Tensor(tf_tensor, name='large_builder_tensor')
        tensor_offset = tensor.to_graph(large_builder, self.buffer_manager)
        self.assertTrue(tensor_offset > 0)

    def test_tensor_deserialization_with_missing_buffer(self):
        # Test deserialization when the buffer corresponding to buffer_idx is missing
        tf_tensor = tf.constant([1, 2, 3], dtype=tf.uint8)
        tensor = Tensor(tf_tensor, name='missing_buffer_tensor')
        tensor_offset = tensor.to_graph(self.builder, self.buffer_manager)
        self.builder.Finish(tensor_offset)
        buffers = []  # Empty buffers list
        io_tensor = IoTensor.Tensor.GetRootAs(self.builder.Output(), 0)

        with self.assertRaises(IndexError):
            Tensor.from_graph(io_tensor, buffers)

if __name__ == '__main__':
    unittest.main()