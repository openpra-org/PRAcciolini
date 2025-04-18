import tensorflow as tf
from typing import List, Optional, Dict, Set

from pracciolini.grammar.canopy.io import (
    SubGraph as IoSubGraph,
    OpCode as IoOpCode,
)
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.buffer_manager import BufferManager
from pracciolini.grammar.canopy.model.operator import Operator, OperatorArgs


class SubGraph:
    """
    Represents a subgraph in the canopy model.

    Attributes:
        name (Optional[str]): Optional name for the subgraph.
        inputs (List[Tensor]): The input tensors to the subgraph.
        outputs (List[Tensor]): The output tensors of the subgraph.
        tensors (List[Tensor]): All tensors used in the subgraph.
        operators (List[Operator]): The operators (operations) in the subgraph.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.tensors: List[Tensor] = []
        self.operators: List[Operator] = []

        # Internal mappings for tensor indices
        self._tensor_to_index: Dict[Tensor, int] = {}
        self._index_to_tensor: Dict[int, Tensor] = {}

        # Dependency mapping: tensor -> set of tensors that depend on it
        self._dependencies: Dict[Tensor, Set[Tensor]] = {}

        # Mapping from OpCode to TensorFlow functions
        self._BITWISE_OPCODES = {
            IoOpCode.OpCode.BITWISE_NOT: tf.bitwise.invert,
            IoOpCode.OpCode.BITWISE_AND: tf.bitwise.bitwise_and,
            IoOpCode.OpCode.BITWISE_OR: tf.bitwise.bitwise_or,
            IoOpCode.OpCode.BITWISE_XOR: tf.bitwise.bitwise_xor,
            IoOpCode.OpCode.RESHAPE: tf.reshape,
            IoOpCode.OpCode.BITWISE_NAND: lambda a, b: tf.bitwise.invert(tf.bitwise.bitwise_and(a, b)),
            IoOpCode.OpCode.BITWISE_NOR: lambda a, b: tf.bitwise.invert(tf.bitwise.bitwise_or(a, b)),
            IoOpCode.OpCode.BITWISE_XNOR: lambda a, b: tf.bitwise.invert(tf.bitwise.bitwise_xor(a, b)),
        }

        # Define the set of supported opcodes
        self._SUPPORTED_OPCODES = self._BITWISE_OPCODES.keys()

    def add_tensor(self, tensor: Tensor) -> int:
        """
        Adds a tensor to the subgraph, if not already present.

        Args:
            tensor (Tensor): The tensor to add.

        Returns:
            int: The index of the tensor in the subgraph's tensors list.
        """
        if tensor in self._tensor_to_index:
            return self._tensor_to_index[tensor]

        tensor_idx = len(self.tensors)
        self.tensors.append(tensor)
        self._tensor_to_index[tensor] = tensor_idx
        self._index_to_tensor[tensor_idx] = tensor
        # Initialize dependencies for the new tensor
        self._dependencies[tensor] = set()
        return tensor_idx

    def register_input(self, tensor: Tensor):
        """
        Registers a tensor as an input to the subgraph.

        Args:
            tensor (Tensor): The tensor to register as input.
        """
        self.add_tensor(tensor)
        if tensor not in self.inputs:
            self.inputs.append(tensor)

    def register_output(self, tensor: Tensor):
        """
        Registers a tensor as an output of the subgraph.

        Args:
            tensor (Tensor): The tensor to register as output.
        """
        self.add_tensor(tensor)
        if tensor not in self.outputs:
            self.outputs.append(tensor)

    def add_operator(
        self,
        opcode: IoOpCode.OpCode,
        input_tensors: List[Tensor],
        output_tensors: List[Tensor],
        args: Optional[OperatorArgs] = None,
        name: Optional[str] = None,
    ):
        """
        Adds an operator (operation) to the subgraph.

        Args:
            opcode (IoOpCode.OpCode): The opcode of the operator.
            input_tensors (List[Tensor]): The input tensors.
            output_tensors (List[Tensor]): The output tensors.
            args (Optional[OperatorArgs]): The operator arguments.
            name (Optional[str]): Optional name of the operator.

        Raises:
            ValueError: If adding the operator creates a circular dependency.
            NotImplementedError: If the opcode is not supported.
        """
        # Validate the opcode
        if opcode not in self._SUPPORTED_OPCODES:
            print(self._SUPPORTED_OPCODES)
            raise NotImplementedError(f"Operator with opcode {opcode} is not implemented")

        # Ensure tensors are added to the subgraph
        input_indices = [self.add_tensor(tensor) for tensor in input_tensors]
        output_indices = [self.add_tensor(tensor) for tensor in output_tensors]

        # Check for circular dependencies
        for output_tensor in output_tensors:
            if self._creates_cycle(output_tensor, input_tensors):
                raise ValueError(
                    f"Adding operator '{name}' creates a circular dependency involving tensor '{output_tensor.name}'."
                )

        # Update dependencies
        for output_tensor in output_tensors:
            self._dependencies[output_tensor] = set()
            for input_tensor in input_tensors:
                self._dependencies[input_tensor].add(output_tensor)

        operator = Operator(
            opcode=opcode,
            inputs=input_indices,
            outputs=output_indices,
            args=args,
            name=name,
        )
        self.operators.append(operator)

    def to_graph(self, builder, buffers: BufferManager) -> int:
        """
        Serializes the subgraph to FlatBuffers using the builder.

        Args:
            builder (flatbuffers.Builder): The FlatBuffers builder.
            buffers (BufferManager): The buffer manager for data buffers.

        Returns:
            int: The offset in the FlatBuffer where the SubGraph is stored.
        """
        # Serialize tensors
        tensor_offsets = []
        for tensor in self.tensors:
            tensor_offset = tensor.to_graph(builder, buffers)
            tensor_offsets.append(tensor_offset)

        # Build tensors vector
        IoSubGraph.SubGraphStartTensorsVector(builder, len(tensor_offsets))
        for offset in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(offset)
        tensors_vector = builder.EndVector(len(tensor_offsets))

        # Serialize inputs
        input_indices = [self._tensor_to_index[tensor] for tensor in self.inputs]
        IoSubGraph.SubGraphStartInputsVector(builder, len(input_indices))
        for idx in reversed(input_indices):
            builder.PrependInt32(idx)
        inputs_vector = builder.EndVector(len(input_indices))

        # Serialize outputs
        output_indices = [self._tensor_to_index[tensor] for tensor in self.outputs]
        IoSubGraph.SubGraphStartOutputsVector(builder, len(output_indices))
        for idx in reversed(output_indices):
            builder.PrependInt32(idx)
        outputs_vector = builder.EndVector(len(output_indices))

        # Serialize operators
        operator_offsets = []
        for operator in self.operators:
            operator_offset = operator.to_graph(builder)
            operator_offsets.append(operator_offset)

        IoSubGraph.SubGraphStartOperatorsVector(builder, len(operator_offsets))
        for offset in reversed(operator_offsets):
            builder.PrependUOffsetTRelative(offset)
        operators_vector = builder.EndVector(len(operator_offsets))

        # Serialize name
        if self.name:
            name_offset = builder.CreateString(self.name)
        else:
            name_offset = None

        # Build SubGraph object
        IoSubGraph.SubGraphStart(builder)
        IoSubGraph.SubGraphAddTensors(builder, tensors_vector)
        IoSubGraph.SubGraphAddInputs(builder, inputs_vector)
        IoSubGraph.SubGraphAddOutputs(builder, outputs_vector)
        IoSubGraph.SubGraphAddOperators(builder, operators_vector)
        if name_offset:
            IoSubGraph.SubGraphAddName(builder, name_offset)
        subgraph_offset = IoSubGraph.SubGraphEnd(builder)

        return subgraph_offset

    @classmethod
    def from_graph(cls, io_subgraph: IoSubGraph, buffers: List[bytes]) -> 'SubGraph':
        """
        Deserializes a SubGraph from a FlatBuffer.

        Args:
            io_subgraph (IoSubGraph): The deserialized SubGraph.
            buffers (List[bytes]): The list of data buffers.

        Returns:
            SubGraph: The SubGraph instance.
        """
        subgraph = cls(name=io_subgraph.Name())

        # Deserialize tensors
        tensor_map = {}  # index to Tensor
        for i in range(io_subgraph.TensorsLength()):
            io_tensor = io_subgraph.Tensors(i)
            tensor = Tensor.from_graph(io_tensor, buffers)
            subgraph.add_tensor(tensor)
            tensor_map[i] = tensor

        # Register inputs
        for i in range(io_subgraph.InputsLength()):
            tensor_idx = io_subgraph.Inputs(i)
            tensor = tensor_map[tensor_idx]
            subgraph.register_input(tensor)

        # Register outputs
        for i in range(io_subgraph.OutputsLength()):
            tensor_idx = io_subgraph.Outputs(i)
            tensor = tensor_map[tensor_idx]
            subgraph.register_output(tensor)

        # Deserialize operators
        for i in range(io_subgraph.OperatorsLength()):
            io_operator = io_subgraph.Operators(i)
            operator = Operator.from_graph(io_operator, tensor_map)
            subgraph.operators.append(operator)

        return subgraph

    def bitwise(self, opcode: IoOpCode, operands: List[Tensor], name: Optional[str] = None) -> Tensor:
        """
        Performs a bitwise operation specified by the opcode on the given tensors and adds it to the subgraph.

        Args:
            opcode (IoOpCode): The opcode of the bitwise operation.
            operands (List[Tensor]): List of input tensors.
            name (Optional[str]): Optional name for the operation.

        Returns:
            Tensor: The resulting tensor.
        """
        # Ensure the opcode is a valid bitwise opcode
        if opcode not in self._BITWISE_OPCODES:
            raise ValueError(f"Opcode {opcode} is not a valid bitwise operation.")

        # Ensure operands are already in the subgraph
        for tensor in operands:
            if tensor not in self._tensor_to_index:
                raise KeyError(f"Input tensor {tensor.name} not found in subgraph.")

        # Get the TensorFlow function for the opcode
        tf_function = self._BITWISE_OPCODES[opcode]

        # Perform the operation
        with tf.name_scope(name or "BitwiseOp"):
            if opcode == IoOpCode.OpCode.BITWISE_NOT:
                if len(operands) != 1:
                    raise ValueError("BITWISE_NOT operation requires exactly one operand.")
                tf_output = tf_function(operands[0].tf_tensor)
            else:
                if len(operands) != 2:
                    raise ValueError(f"Operation with opcode {opcode} requires exactly two operands.")
                tf_output = tf_function(operands[0].tf_tensor, operands[1].tf_tensor)

        # Create output tensor
        output_tensor = Tensor(tf_output, name=name)

        # Add the operation to the subgraph
        self.add_operator(
            opcode=opcode,
            input_tensors=operands,
            output_tensors=[output_tensor],
            args=None,
            name=name,
        )

        return output_tensor

    # Optional helper methods for convenience
    def bitwise_and(self, a: Tensor, b: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_AND, [a, b], name)

    def bitwise_or(self, a: Tensor, b: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_OR, [a, b], name)

    def bitwise_xor(self, a: Tensor, b: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_XOR, [a, b], name)

    def bitwise_not(self, a: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_NOT, [a], name)

    def bitwise_nand(self, a: Tensor, b: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_NAND, [a, b], name)

    def bitwise_nor(self, a: Tensor, b: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_NOR, [a, b], name)

    def bitwise_xnor(self, a: Tensor, b: Tensor, name: Optional[str] = None) -> Tensor:
        return self.bitwise(IoOpCode.OpCode.BITWISE_XNOR, [a, b], name)

    # Additional methods for other operations can be implemented similarly

    def execute(self, feed_dict: Dict[Tensor, tf.Tensor]) -> Dict[Tensor, tf.Tensor]:
        """
        Executes the subgraph given input values.

        Args:
            feed_dict (Dict[Tensor, tf.Tensor]): Mapping from input Tensors to their values.

        Returns:
            Dict[Tensor, tf.Tensor]: Mapping from output Tensors to their evaluated values.
        """
        # Prepare a mapping from Tensor to their actual values
        tensor_values = feed_dict.copy()

        # Evaluate the outputs
        output_values = {}
        for output_tensor in self.outputs:
            # Evaluate the output tensor using its tf.Tensor object and the tensor_values mapping
            value = self._evaluate_tensor(output_tensor, tensor_values)
            output_values[output_tensor] = value
        return output_values

    def _evaluate_tensor(self, tensor: Tensor, tensor_values: Dict[Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Recursively evaluates the tensor by evaluating its dependencies.

        Args:
            tensor (Tensor): The tensor to evaluate.
            tensor_values (Dict[Tensor, tf.Tensor]): Mapping from tensors to their evaluated values.

        Returns:
            tf.Tensor: The evaluated tensor value.
        """
        if tensor in tensor_values:
            return tensor_values[tensor]
        # Find the operator that produces this tensor
        for operator in self.operators:
            if tensor in [self._index_to_tensor[idx] for idx in operator.outputs]:
                # Evaluate the operator's inputs recursively
                input_tensors = [self._index_to_tensor[idx] for idx in operator.inputs]
                input_values = [self._evaluate_tensor(t, tensor_values) for t in input_tensors]

                # Perform the operation
                value = self._execute_operator(operator, input_values)
                tensor_values[tensor] = value
                return value
        # Check if tensor is a constant (has a tf_tensor but is not an input or output of any operator)
        if tensor.tf_tensor is not None:
            tensor_values[tensor] = tensor.tf_tensor
            return tensor.tf_tensor
        raise ValueError(f"Cannot find operator that produces tensor {tensor.name}")

    def _execute_operator(self, operator: Operator, input_values: List[tf.Tensor]) -> tf.Tensor:
        """
        Executes the operator with the given input values.

        Args:
            operator (Operator): The operator to execute.
            input_values (List[tf.Tensor]): The evaluated input values.

        Returns:
            tf.Tensor: The result of the operator.
        """
        opcode = operator.opcode

        # Check if the opcode is a bitwise operation
        if opcode in self._BITWISE_OPCODES:
            tf_function = self._BITWISE_OPCODES[opcode]
            if opcode == IoOpCode.OpCode.BITWISE_NOT:
                if len(input_values) != 1:
                    raise ValueError("BITWISE_NOT operation requires exactly one input.")
                return tf_function(input_values[0])
            else:
                if len(input_values) != 2:
                    raise ValueError(f"Operation with opcode {opcode} requires exactly two inputs.")
                return tf_function(input_values[0], input_values[1])
        else:
            raise NotImplementedError(f"Operator with opcode {opcode} is not implemented")

    def execute_function(self):
        """
        Returns a callable function that executes the subgraph with inputs as arguments.

        The function expects inputs in the order of the subgraph's inputs.

        Returns:
            Callable: A function that when called with input tensors returns output tensors.
        """

        @tf.function
        def func(*inputs):
            feed_dict = dict(zip(self.inputs, inputs))
            outputs = self.execute(feed_dict)
            return tuple(outputs[tensor] for tensor in self.outputs)

        return func

    def _creates_cycle(self, target_tensor: Tensor, input_tensors: List[Tensor]) -> bool:
        """
        Checks whether adding an operator with the given inputs and target output tensor
        would create a circular dependency.

        Args:
            target_tensor (Tensor): The output tensor of the operator being added.
            input_tensors (List[Tensor]): The input tensors to the operator.

        Returns:
            bool: True if a cycle is detected, False otherwise.
        """
        visited = set()
        stack = set()

        def visit(tensor):
            if tensor in stack:
                return True  # Cycle detected
            if tensor in visited:
                return False
            visited.add(tensor)
            stack.add(tensor)
            for dependent_tensor in self._dependencies.get(tensor, []):
                if visit(dependent_tensor):
                    return True
            stack.remove(tensor)
            return False

        # Temporarily add the new dependencies and check for cycles
        original_dependencies = {}
        for input_tensor in input_tensors:
            original_dependencies.setdefault(input_tensor, set()).update(
                self._dependencies.get(input_tensor, set())
            )
            self._dependencies.setdefault(input_tensor, set()).add(target_tensor)

        has_cycle = visit(target_tensor)

        # Restore original dependencies
        for input_tensor in input_tensors:
            self._dependencies[input_tensor] = original_dependencies[input_tensor]

        return has_cycle

    def to_tensorflow_graph(self) -> tf.Graph:
        """
        Constructs a TensorFlow graph by traversing this SubGraph,
        collecting the tensors and operations.

        Returns:
            tf.Graph: A TensorFlow graph that represents the subgraph.
        """
        graph = tf.Graph()
        with graph.as_default():
            # Disable eager execution to build a static graph
            tf.compat.v1.disable_eager_execution()

            # Create a mapping from our model tensors to the TensorFlow tensors in the graph
            tensor_mapping: Dict[Tensor, tf.Tensor] = {}

            # Map input tensors to placeholders
            for tensor in self.inputs:
                tf_tensor = tf.compat.v1.placeholder(
                    dtype=tensor.tf_tensor.dtype,
                    shape=tensor.tf_tensor.shape,
                    name=tensor.name or f"input_{self._tensor_to_index[tensor]}"
                )
                tensor_mapping[tensor] = tf_tensor

            # Process operators in the order they were added
            for operator in self.operators:
                # Get input tf.Tensors from the tensor mapping
                input_tensors = [tensor_mapping[self._index_to_tensor[idx]] for idx in operator.inputs]

                # Get the TensorFlow function corresponding to the opcode
                opcode = operator.opcode
                if opcode in self._BITWISE_OPCODES:
                    tf_function = self._BITWISE_OPCODES[opcode]

                    # Prepare arguments
                    if opcode == IoOpCode.OpCode.BITWISE_NOT:
                        if len(input_tensors) != 1:
                            raise ValueError("BITWISE_NOT operation requires exactly one operand.")
                        result = tf_function(input_tensors[0], name=operator.name)
                    else:
                        if len(input_tensors) != 2:
                            raise ValueError(f"Operation with opcode {opcode} requires exactly two operands.")
                        result = tf_function(input_tensors[0], input_tensors[1], name=operator.name)
                else:
                    raise NotImplementedError(f"Operator with opcode {opcode} is not implemented in to_tensorflow_graph")

                # Map output tensors to the result
                if len(operator.outputs) != 1:
                    raise ValueError("Only operators with a single output are supported in to_tensorflow_graph")
                output_tensor = self._index_to_tensor[operator.outputs[0]]
                tensor_mapping[output_tensor] = result

            # Map output tensors
            for tensor in self.outputs:
                if tensor not in tensor_mapping:
                    # The tensor might be a direct input or a constant
                    if tensor in self.inputs:
                        tensor_mapping[tensor] = tensor_mapping[tensor]
                    elif tensor.tf_tensor is not None:
                        tensor_mapping[tensor] = tensor.tf_tensor
                    else:
                        raise ValueError(f"Output tensor {tensor.name} not found in tensor mapping")

            # Optionally, add output tensors to a collection for easy retrieval
            output_tensors = [tensor_mapping[tensor] for tensor in self.outputs]
            for i, out_tensor in enumerate(output_tensors):
                tf.identity(out_tensor, name=f"output_{i}")

        return graph