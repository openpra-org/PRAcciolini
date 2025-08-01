import flatbuffers

from pracciolini.grammar.canopy.io import Buffer

class BufferManager:
    """
    Manages the buffers for the model.
    """
    def __init__(self):
        self._buffers = []  # List of data buffers
        self._buffer_offsets = []  # List of buffer offsets in FlatBuffers

        # Ensure the first buffer is an empty buffer as per the schema requirement
        self.add_empty_buffer()

    def add_empty_buffer(self):
        """
        Adds an empty buffer as the first buffer.
        """
        self._buffers.append(b'')
        self._buffer_offsets.append(0)  # Placeholder, will be updated during serialization

    def add_buffer(self, data_bytes: bytes) -> int:
        """
        Adds a buffer and returns the buffer index.

        Args:
            data_bytes (bytes): The data to be stored in the buffer.

        Returns:
            int: The buffer index.
        """
        if data_bytes is None:
            raise TypeError("buffer bytes cannot be None")

        if not isinstance(data_bytes, bytes):
            raise TypeError("buffer has to be of type bytes")

        # Store data (for lookup)
        self._buffers.append(data_bytes)

        buffer_idx = len(self._buffers) - 1
        return buffer_idx

    def serialize_buffers(self, builder: flatbuffers.Builder):
        """
        Serializes all the buffers to FlatBuffers.

        Args:
            builder (flatbuffers.Builder): The FlatBuffers builder.
        """
        # Serialize buffers in reverse order
        buffer_offsets = []
        for data_bytes in reversed(self._buffers):
            if data_bytes:
                data_vector = builder.CreateByteVector(data_bytes)
            else:
                data_vector = 0  # Empty buffer

            Buffer.Start(builder)
            if data_vector:
                Buffer.AddData(builder, data_vector)
            buffer_offset = Buffer.End(builder)
            buffer_offsets.append(buffer_offset)
        # The offsets are in reverse order, so reverse them back
        buffer_offsets.reverse()
        self._buffer_offsets = buffer_offsets

    def get_buffer_offsets(self):
        """
        Returns the list of buffer offsets for building the buffers table.
        """
        return self._buffer_offsets

    def get_buffers(self):
        """
        Returns the list of buffer data bytes.
        """
        return self._buffers
