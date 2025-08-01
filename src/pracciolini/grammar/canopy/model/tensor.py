import tensorflow as tf
import numpy as np
from typing import Optional, List

from pracciolini.grammar.canopy.io import Tensor as IoTensor
from pracciolini.grammar.canopy.io import TensorType as IoTensorType
from pracciolini.grammar.canopy.model.buffer_manager import BufferManager


class Tensor:
    """
    Represents a tensor in the canopy model.

    Attributes:
        name (Optional[str]): Optional name for the tensor, for debugging purposes.
        tf_tensor (tf.Tensor): The underlying TensorFlow Tensor.
        shape (List[int]): The shape of the tensor, where -1 represents dynamic dimensions.
    """

    def __init__(self, tensor: tf.Tensor|tf.Variable, name: Optional[str] = None, shape: Optional[List[int]] = None):
        if not isinstance(tensor, tf.Tensor):
            if not isinstance(tensor, tf.Variable):
                raise TypeError("tensor must be a TensorFlow Tensor")
        self.tf_tensor: tf.Tensor | tf.Variable = tensor
        self.name = name
        if shape is not None:
            # Replace None with -1 to represent dynamic dimensions
            self.shape = [dim if dim is not None else -1 for dim in shape]
        else:
            # Use tensor's shape, handle dynamic dimensions
            self.shape = [
                dim if dim is not None else -1 for dim in self.tf_tensor.shape.as_list()
            ]

    def to_graph(self, builder, buffers: BufferManager) -> int:
        """
        Serializes the model Tensor to FlatBuffers using builder.

        Args:
            builder (flatbuffers.Builder): The FlatBuffers builder to serialize the data.
            buffers (BufferManager): BufferManager object to handle data buffers.

        Returns:
            int: The offset in the FlatBuffer where the Tensor is stored.
        """
        # Map TensorFlow dtype to TensorType
        tensor_type = self._map_tf_dtype_to_tensortype(self.tf_tensor.dtype)

        # Serialize shape
        shape_vector = self._build_shape_vector(builder, self.shape)

        # Get tensor data as bytes and add to buffers
        tensor_data = self._get_tensor_data_bytes()

        # Add buffer and get buffer_idx
        buffer_idx = buffers.add_buffer(tensor_data)

        # Serialize name
        if self.name:
            name_offset = builder.CreateString(self.name)
        else:
            name_offset = None

        # Build the Tensor object
        IoTensor.TensorStart(builder)
        IoTensor.TensorAddShape(builder, shape_vector)
        IoTensor.TensorAddType(builder, tensor_type)
        IoTensor.TensorAddBufferIdx(builder, buffer_idx)
        if name_offset is not None:
            IoTensor.TensorAddName(builder, name_offset)
        tensor_offset = IoTensor.TensorEnd(builder)

        return tensor_offset

    @classmethod
    def from_graph(cls, io_tensor: IoTensor, buffers: List[bytes]) -> 'Tensor':
        """
        Creates a model Tensor from a deserialized FlatBuffer Tensor.

        Args:
            io_tensor (canopy.io.Tensor): The deserialized Tensor from FlatBuffers.
            buffers (list of bytes): List of data buffers corresponding to the buffers table.

        Returns:
            Tensor: The model Tensor instance.
        """
        # Get shape
        shape = [io_tensor.Shape(i) for i in range(io_tensor.ShapeLength())]

        # For numpy reshape, represent dynamic dimensions with -1
        shape_for_reshape = [dim if dim != -1 else -1 for dim in shape]

        # Get dtype
        tensor_type = io_tensor.Type()
        dtype = cls._map_tensortype_to_tf_dtype(tensor_type)

        # Get buffer index
        buffer_idx = io_tensor.BufferIdx()

        # Get data from buffers
        if buffer_idx >= len(buffers):
            raise IndexError(f"Buffer index {buffer_idx} out of bounds")
        tensor_data_bytes = buffers[buffer_idx]

        # Convert bytes to numpy array
        tensor_data = np.frombuffer(tensor_data_bytes, dtype=dtype.as_numpy_dtype)
        # Reshape numpy array
        try:
            tensor_data = tensor_data.reshape(shape_for_reshape)
        except ValueError as e:
            raise ValueError(f"Cannot reshape tensor data to shape {shape}: {e}")

        # Create TensorFlow Tensor
        tensor = tf.convert_to_tensor(tensor_data, dtype=dtype)

        # Get name
        name = io_tensor.Name()

        return cls(tensor, name)

    def _get_tensor_data_bytes(self) -> bytes:
        """
        Returns the tensor data as bytes.
        """
        # Ensure that the tensor is evaluated and get numpy data
        if not self.tf_tensor.dtype.is_numpy_compatible:
            raise TypeError(f"TensorFlow dtype {self.tf_tensor.dtype} is not numpy compatible")
        tensor_data = self.tf_tensor.numpy()
        return tensor_data.tobytes()

    @staticmethod
    def _map_tf_dtype_to_tensortype(dtype):
        """
        Maps TensorFlow dtype to TensorType enum.
        """
        if dtype == tf.uint8:
            return IoTensorType.TensorType.UINT8
        elif dtype == tf.uint16:
            return IoTensorType.TensorType.UINT16
        elif dtype == tf.uint32:
            return IoTensorType.TensorType.UINT32
        elif dtype == tf.uint64:
            return IoTensorType.TensorType.UINT64
        elif dtype == tf.float16:
            return IoTensorType.TensorType.FLOAT16
        elif dtype == tf.float32:
            return IoTensorType.TensorType.FLOAT32
        elif dtype == tf.float64:
            return IoTensorType.TensorType.FLOAT64
        else:
            raise ValueError(f"Unsupported data type {dtype}")

    @staticmethod
    def _map_tensortype_to_tf_dtype(tensortype):
        """
        Maps TensorType enum to TensorFlow dtype.
        """
        if tensortype == IoTensorType.TensorType.UINT8:
            return tf.uint8
        elif tensortype == IoTensorType.TensorType.UINT16:
            return tf.uint16
        elif tensortype == IoTensorType.TensorType.UINT32:
            return tf.uint32
        elif tensortype == IoTensorType.TensorType.UINT64:
            return tf.uint64
        elif tensortype == IoTensorType.TensorType.FLOAT16:
            return tf.float16
        elif tensortype == IoTensorType.TensorType.FLOAT32:
            return tf.float32
        elif tensortype == IoTensorType.TensorType.FLOAT64:
            return tf.float64
        else:
            raise ValueError(f"Unsupported TensorType {tensortype}")

    @staticmethod
    def _build_shape_vector(builder, shape) -> int:
        """
        Builds the shape vector for FlatBuffers.

        Args:
            builder (flatbuffers.Builder): The FlatBuffers builder.
            shape (List[int]): The tensor shape.

        Returns:
            int: The offset for the shape vector.
        """
        IoTensor.TensorStartShapeVector(builder, len(shape))
        for dim in reversed(shape):
            builder.PrependInt32(dim)
        return builder.EndVector(len(shape))
