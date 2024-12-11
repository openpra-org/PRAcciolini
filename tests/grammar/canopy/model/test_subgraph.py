import unittest

import numpy as np
import tensorflow as tf
import flatbuffers

from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph
from pracciolini.grammar.canopy.model.buffer_manager import BufferManager
from pracciolini.grammar.canopy.io import (
    OpCode as IoOpCode,
    DAGs as IoDAGs,
    SubGraph as IoSubGraph,
)

class SubGraphCoreTests(unittest.TestCase):
    def setUp(self):
        # Set up common tensors and subgraph for tests
        self.tensor_a = Tensor(tf.constant([1, 2, 3], dtype=tf.uint8), name="tensor_a")
        self.tensor_b = Tensor(tf.constant([4, 5, 6], dtype=tf.uint8), name="tensor_b")
        self.tensor_c = Tensor(tf.constant([7, 8, 9], dtype=tf.uint8), name="tensor_c")
        self.subgraph = SubGraph(name="TestSubGraph")

    def test_add_tensor(self):
        """
        Test that adding tensors to the subgraph works correctly.
        """
        idx_a = self.subgraph.add_tensor(self.tensor_a)
        self.subgraph.add_tensor(self.tensor_b)
        # Adding the same tensor should not duplicate it
        idx_a_dup = self.subgraph.add_tensor(self.tensor_a)
        self.assertEqual(idx_a, idx_a_dup, "Duplicate tensor should return the same index")
        self.assertEqual(len(self.subgraph.tensors), 2, "Subgraph should contain two tensors")

    def test_register_input_output(self):
        """
        Test registering tensors as inputs and outputs.
        """
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_output(self.tensor_b)
        self.assertIn(self.tensor_a, self.subgraph.inputs, "tensor_a should be in inputs")
        self.assertIn(self.tensor_b, self.subgraph.outputs, "tensor_b should be in outputs")
        # Ensure tensors are added to the tensors list
        self.assertIn(self.tensor_a, self.subgraph.tensors, "tensor_a should be in tensors")
        self.assertIn(self.tensor_b, self.subgraph.tensors, "tensor_b should be in tensors")

    def test_add_operator(self):
        """
        Test that adding operators works correctly.
        """
        # Add tensors to subgraph
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)
        # Define an output tensor
        result_tensor = Tensor(tf.bitwise.bitwise_and(self.tensor_a.tf_tensor, self.tensor_b.tf_tensor), name="result_tensor")
        # Add operator
        self.subgraph.add_operator(
            opcode=IoOpCode.OpCode.BITWISE_AND,
            input_tensors=[self.tensor_a, self.tensor_b],
            output_tensors=[result_tensor],
            args=None,
            name="BitwiseAndOp",
        )
        # Check that operator is added
        self.assertEqual(len(self.subgraph.operators), 1, "Subgraph should have one operator")
        operator = self.subgraph.operators[0]
        self.assertEqual(operator.opcode, IoOpCode.OpCode.BITWISE_AND, "Operator opcode should be BITWISE_AND")
        # Check that output tensor is added to tensors
        self.assertIn(result_tensor, self.subgraph.tensors, "Result tensor should be in tensors")

    def test_bitwise_and_method(self):
        """
        Test the bitwise_and method of SubGraph.
        """
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)
        # Use bitwise_and method
        result_tensor = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="BitwiseAndOp")
        # Verify the operator was added
        self.assertEqual(len(self.subgraph.operators), 1, "Subgraph should have one operator after bitwise_and")
        operator = self.subgraph.operators[0]
        self.assertEqual(operator.opcode, IoOpCode.OpCode.BITWISE_AND, "Operator opcode should be BITWISE_AND")
        # Verify result tensor
        expected_result = tf.bitwise.bitwise_and(self.tensor_a.tf_tensor, self.tensor_b.tf_tensor)
        self.assertTrue(tf.reduce_all(tf.equal(result_tensor.tf_tensor, expected_result)), "Result tensor should match expected value")

    def test_serialization_deserialization(self):
        """
        Test serialization and deserialization of the subgraph.
        """
        # Set up subgraph with tensors and operator
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)
        result_tensor = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="BitwiseAndOp")
        self.subgraph.register_output(result_tensor)

        # Serialize
        builder = flatbuffers.Builder(1024)
        buffers = BufferManager()

        # Serialize the subgraph
        subgraph_offset = self.subgraph.to_graph(builder, buffers)

        # Serialize buffers
        buffers.serialize_buffers(builder)
        buffer_offsets = buffers.get_buffer_offsets()

        # Build buffers vector
        IoDAGs.DAGsStartBuffersVector(builder, len(buffer_offsets))
        for offset in reversed(buffer_offsets):
            builder.PrependUOffsetTRelative(offset)
        buffers_vector = builder.EndVector(len(buffer_offsets))

        # Build subgraphs vector
        IoDAGs.DAGsStartSubgraphsVector(builder, 1)
        builder.PrependUOffsetTRelative(subgraph_offset)
        subgraphs_vector = builder.EndVector(1)

        # Create strings before starting the DAGs object
        name_offset = builder.CreateString("TestModel")
        description_offset = builder.CreateString("Test Description")

        # Finish DAGs
        IoDAGs.DAGsStart(builder)
        IoDAGs.DAGsAddSubgraphs(builder, subgraphs_vector)
        IoDAGs.DAGsAddBuffers(builder, buffers_vector)
        IoDAGs.DAGsAddName(builder, name_offset)
        IoDAGs.DAGsAddDescription(builder, description_offset)
        dags_offset = IoDAGs.DAGsEnd(builder)
        builder.Finish(dags_offset)

        buf = builder.Output()

        # Deserialize
        io_dags = IoDAGs.DAGs.GetRootAs(buf, 0)
        io_subgraph = io_dags.Subgraphs(0)
        buffer_data = buffers.get_buffers()
        deserialized_subgraph = SubGraph.from_graph(io_subgraph, buffer_data)

        # Verify deserialized subgraph
        self.assertEqual(deserialized_subgraph.name.decode('utf-8'), self.subgraph.name,"Subgraph name should match after deserialization")
        self.assertEqual(len(deserialized_subgraph.tensors), len(self.subgraph.tensors),"Number of tensors should match after deserialization")
        self.assertEqual(len(deserialized_subgraph.operators), len(self.subgraph.operators),"Number of operators should match after deserialization")

        # Verify tensors
        for original_tensor, deserialized_tensor in zip(self.subgraph.tensors, deserialized_subgraph.tensors):
            self.assertEqual(original_tensor.name, deserialized_tensor.name.decode('utf-8'),"Tensor names should match after deserialization")
            self.assertTrue(
                tf.reduce_all(tf.equal(original_tensor.tf_tensor, deserialized_tensor.tf_tensor)),
                "Tensor values should match after deserialization",
            )

        # Verify operators
        for original_op, deserialized_op in zip(self.subgraph.operators, deserialized_subgraph.operators):
            self.assertEqual(original_op.opcode, deserialized_op.opcode,"Operator opcodes should match after deserialization")
            self.assertEqual(original_op.inputs, deserialized_op.inputs,"Operator inputs should match after deserialization")
            self.assertEqual(original_op.outputs, deserialized_op.outputs,"Operator outputs should match after deserialization")

    def test_operator_args(self):
        """
        Test adding an operator with arguments.
        """
        # Add tensors to subgraph
        self.subgraph.register_input(self.tensor_a)
        # Output tensor
        reshaped_tensor = Tensor(
            tf.reshape(self.tensor_a.tf_tensor, [3, 1]), name="reshaped_tensor"
        )
        # Reshape operator arguments
        from pracciolini.grammar.canopy.model.operator import ReshapeOperatorArgs
        reshape_args = ReshapeOperatorArgs(new_shape=[3, 1])
        # Add operator
        self.subgraph.add_operator(
            opcode=IoOpCode.OpCode.RESHAPE,
            input_tensors=[self.tensor_a],
            output_tensors=[reshaped_tensor],
            args=reshape_args,
            name="ReshapeOp",
        )
        # Verify operator
        self.assertEqual(len(self.subgraph.operators), 1, "Subgraph should have one operator")
        operator = self.subgraph.operators[0]
        self.assertEqual(operator.opcode, IoOpCode.OpCode.RESHAPE, "Operator opcode should be RESHAPE")
        self.assertIsInstance(operator.args, ReshapeOperatorArgs, "Operator arguments should be ReshapeOperatorArgs")
        self.assertEqual(operator.args.new_shape, [3, 1], "Operator arguments should have new shape [3, 1]")

    def test_multiple_operations(self):
        """
        Test adding multiple operations and ensuring indices and mappings are consistent.
        """
        # Register tensors
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)
        # First operation
        result_tensor1 = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="BitwiseAndOp1")
        # Second operation
        result_tensor2 = self.subgraph.bitwise_and(result_tensor1, self.tensor_b, name="BitwiseAndOp2")
        self.subgraph.register_output(result_tensor2)
        # Verify tensors
        self.assertEqual(len(self.subgraph.tensors), 4, "Subgraph should have four tensors")
        # Verify operators
        self.assertEqual(len(self.subgraph.operators), 2, "Subgraph should have two operators")
        # Verify indices are correct
        idx_tensor_a = self.subgraph._tensor_to_index[self.tensor_a]
        idx_tensor_b = self.subgraph._tensor_to_index[self.tensor_b]
        idx_result1 = self.subgraph._tensor_to_index[result_tensor1]
        idx_result2 = self.subgraph._tensor_to_index[result_tensor2]
        operator1 = self.subgraph.operators[0]
        operator2 = self.subgraph.operators[1]
        self.assertEqual(operator1.inputs, [idx_tensor_a, idx_tensor_b], "Operator1 inputs should be correct")
        self.assertEqual(operator1.outputs, [idx_result1], "Operator1 outputs should be correct")
        self.assertEqual(operator2.inputs, [idx_result1, idx_tensor_b], "Operator2 inputs should be correct")
        self.assertEqual(operator2.outputs, [idx_result2], "Operator2 outputs should be correct")

    def test_mismatched_tensor_dtypes(self):
        """
        Test adding operators with tensors of mismatched data types, which should raise errors.
        """
        # Create tensors of different dtypes
        tensor_float = Tensor(tf.constant([1.0, 2.0, 3.0], dtype=tf.float32), name="tensor_float")
        tensor_int = Tensor(tf.constant([1, 2, 3], dtype=tf.uint8), name="tensor_int")
        self.subgraph.register_input(tensor_float)
        self.subgraph.register_input(tensor_int)
        # Attempt to perform bitwise AND on tensors of different types
        with self.assertRaises(tf.errors.InvalidArgumentError, msg="Bitwise operation on mismatched dtypes should raise TypeError"):
            self.subgraph.bitwise_and(tensor_float, tensor_int, name="InvalidBitwiseOp")

    def test_operator_with_args_serialization(self):
        """
        Test serialization and deserialization of an operator with arguments.
        """
        # Add tensors to subgraph
        self.subgraph.register_input(self.tensor_a)
        # Output tensor
        reshaped_tensor = Tensor(
            tf.reshape(self.tensor_a.tf_tensor, [3, 1]), name="reshaped_tensor"
        )
        # Reshape operator arguments
        from pracciolini.grammar.canopy.model.operator import ReshapeOperatorArgs
        reshape_args = ReshapeOperatorArgs(new_shape=[3, 1])
        # Add operator
        self.subgraph.add_operator(
            opcode=IoOpCode.OpCode.RESHAPE,
            input_tensors=[self.tensor_a],
            output_tensors=[reshaped_tensor],
            args=reshape_args,
            name="ReshapeOp",
        )
        # Serialize
        builder = flatbuffers.Builder(1024)
        buffers = BufferManager()
        subgraph_offset = self.subgraph.to_graph(builder, buffers)
        builder.Finish(subgraph_offset)
        buf = builder.Output()
        # Deserialize
        io_subgraph = IoSubGraph.SubGraph.GetRootAs(buf, 0)
        buffer_data = buffers.get_buffers()
        deserialized_subgraph = SubGraph.from_graph(io_subgraph, buffer_data)
        # Verify operator arguments
        operator = deserialized_subgraph.operators[0]
        self.assertIsInstance(operator.args, ReshapeOperatorArgs, "Deserialized operator arguments should be ReshapeOperatorArgs")
        self.assertEqual(operator.args.new_shape, [3, 1], "Deserialized reshape arguments should have new shape [3, 1]")

    # Additional test methods can be added to cover more scenarios

    def test_execute_subgraph(self):
        """
        Test executing the subgraph using execute().
        """
        # Register inputs and outputs
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)

        # Use the subgraph's method to add BITWISE_AND operation
        output_tensor = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="AndOp")
        self.subgraph.register_output(output_tensor)

        # Provide input values
        input_values = {
            self.tensor_a: tf.constant([1, 0, 1], dtype=tf.uint8),
            self.tensor_b: tf.constant([1, 1, 0], dtype=tf.uint8)
        }

        # Execute the subgraph
        output_values = self.subgraph.execute(input_values)

        # Check the output
        expected_output = tf.bitwise.bitwise_and(input_values[self.tensor_a], input_values[self.tensor_b])
        actual_output = output_values[output_tensor]

        self.assertTrue(tf.reduce_all(tf.equal(actual_output, expected_output)), "Output does not match expected value")

    def test_execute_function(self):
        """
        Test executing the subgraph as a callable function.
        """
        # Register inputs and outputs
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)

        # Use the subgraph's method to add BITWISE_OR operation
        output_tensor = self.subgraph.bitwise_or(self.tensor_a, self.tensor_b, name="OrOp")
        self.subgraph.register_output(output_tensor)

        # Define the inputs
        input_a = tf.constant([0, 0, 1], dtype=tf.uint8)
        input_b = tf.constant([0, 1, 1], dtype=tf.uint8)

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_a, input_b)

        # The result is a tuple of outputs
        actual_output = result[0]
        expected_output = tf.bitwise.bitwise_or(input_a, input_b)

        self.assertTrue(tf.reduce_all(tf.equal(actual_output, expected_output)), "Function output does not match expected value")

    def test_chain_operations(self):
        """
        Test executing a subgraph with a chain of bitwise operations.
        """
        # Register inputs
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)
        self.subgraph.register_input(self.tensor_c)

        # Perform BITWISE_AND between tensor_a and tensor_b
        and_output = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="AndOp")
        # Perform BITWISE_XOR between the result and tensor_c
        xor_output = self.subgraph.bitwise_xor(and_output, self.tensor_c, name="XorOp")
        # Perform BITWISE_NOT on the result
        not_output = self.subgraph.bitwise_not(xor_output, name="NotOp")

        # Register final output
        self.subgraph.register_output(not_output)

        # Define the inputs
        input_a = tf.constant([5, 12, 7], dtype=tf.uint32)
        input_b = tf.constant([3, 6, 14], dtype=tf.uint32)
        input_c = tf.constant([9, 10, 11], dtype=tf.uint32)

        # Expected computations
        and_result = tf.bitwise.bitwise_and(input_a, input_b)
        xor_result = tf.bitwise.bitwise_xor(and_result, input_c)
        expected_output = tf.bitwise.invert(xor_result)

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_a, input_b, input_c)

        # The result is a tuple of outputs
        actual_output = result[0]

        self.assertTrue(
            tf.reduce_all(tf.equal(actual_output, expected_output)),
            "Chain operation output does not match expected value"
        )

    def test_parallel_operations(self):
        """
        Test executing a subgraph with parallel bitwise operations.
        """
        # Register inputs
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)

        # Perform BITWISE_OR and BITWISE_AND in parallel
        or_output = self.subgraph.bitwise_or(self.tensor_a, self.tensor_b, name="OrOp")
        and_output = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="AndOp")
        # Perform BITWISE_XOR on the results
        xor_output = self.subgraph.bitwise_xor(or_output, and_output, name="XorOp")

        # Register final output
        self.subgraph.register_output(xor_output)

        # Define the inputs
        input_a = tf.constant([0xFFFF0000, 0x12345678], dtype=tf.uint32)
        input_b = tf.constant([0x0000FFFF, 0x87654321], dtype=tf.uint32)

        # Expected computations
        or_result = tf.bitwise.bitwise_or(input_a, input_b)
        and_result = tf.bitwise.bitwise_and(input_a, input_b)
        expected_output = tf.bitwise.bitwise_xor(or_result, and_result)

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_a, input_b)

        # The result is a tuple of outputs
        actual_output = result[0]

        self.assertTrue(
            tf.reduce_all(tf.equal(actual_output, expected_output)),
            "Parallel operation output does not match expected value"
        )

    def test_large_tensors(self):
        """
        Test executing a subgraph with large input tensors.
        """
        # Create large input tensors
        size = int(1e6)  # 1 million elements
        input_data_a = tf.random.uniform([size], minval=0, maxval=tf.int32.max, dtype=tf.int32)
        input_data_b = tf.random.uniform([size], minval=0, maxval=tf.int32.max, dtype=tf.int32)

        large_tensor_a = Tensor(input_data_a, name="large_tensor_a")
        large_tensor_b = Tensor(input_data_b, name="large_tensor_b")

        # Reset subgraph for this test
        self.subgraph = SubGraph(name="LargeTensorSubGraph")
        self.subgraph.register_input(large_tensor_a)
        self.subgraph.register_input(large_tensor_b)

        # Perform BITWISE_XOR operation
        xor_output = self.subgraph.bitwise_xor(large_tensor_a, large_tensor_b, name="XorOp")
        # Perform BITWISE_NOT on the result
        not_output = self.subgraph.bitwise_not(xor_output, name="NotOp")

        # Register final output
        self.subgraph.register_output(not_output)

        # Expected computations
        expected_xor = tf.bitwise.bitwise_xor(input_data_a, input_data_b)
        expected_output = tf.bitwise.invert(expected_xor)

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_data_a, input_data_b)

        # The result is a tuple of outputs
        actual_output = result[0]

        # Due to the size, we check for equality using numpy
        actual_output_np = actual_output.numpy()
        expected_output_np = expected_output.numpy()

        self.assertTrue(
            np.array_equal(actual_output_np, expected_output_np),
            "Large tensor operation output does not match expected value"
        )

    def test_mixed_operations(self):
        """
        Test executing a subgraph with mixed bitwise operations on larger tensors.
        """
        # Create input tensors with medium size
        size = int(1e6)  # 1 mil
        input_data_a = tf.constant(np.random.randint(0, 0xFFFFFFFF, size, dtype=np.uint32))
        input_data_b = tf.constant(np.random.randint(0, 0xFFFFFFFF, size, dtype=np.uint32))
        input_data_c = tf.constant(np.random.randint(0, 0xFFFFFFFF, size, dtype=np.uint32))
        input_data_d = tf.constant(np.random.randint(0, 0xFFFFFFFF, size, dtype=np.uint32))

        tensor_a = Tensor(input_data_a, name="tensor_a")
        tensor_b = Tensor(input_data_b, name="tensor_b")
        tensor_c = Tensor(input_data_c, name="tensor_c")
        tensor_d = Tensor(input_data_d, name="tensor_d")

        # Reset subgraph for this test
        self.subgraph = SubGraph(name="MixedOperationsSubGraph")
        self.subgraph.register_input(tensor_a)
        self.subgraph.register_input(tensor_b)
        self.subgraph.register_input(tensor_c)
        self.subgraph.register_input(tensor_d)

        # Perform BITWISE_AND between tensor_a and tensor_b
        and_output = self.subgraph.bitwise_and(tensor_a, tensor_b, name="AndOp")
        # Perform BITWISE_OR between tensor_c and tensor_d
        or_output = self.subgraph.bitwise_or(tensor_c, tensor_d, name="OrOp")
        # Perform BITWISE_XOR between the results
        xor_output = self.subgraph.bitwise_xor(and_output, or_output, name="XorOp")
        # Perform BITWISE_NOT on the final result
        not_output = self.subgraph.bitwise_not(xor_output, name="NotOp")

        # Register final output
        self.subgraph.register_output(not_output)

        # Expected computations
        expected_and = tf.bitwise.bitwise_and(input_data_a, input_data_b)
        expected_or = tf.bitwise.bitwise_or(input_data_c, input_data_d)
        expected_xor = tf.bitwise.bitwise_xor(expected_and, expected_or)
        expected_output = tf.bitwise.invert(expected_xor)

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_data_a, input_data_b, input_data_c, input_data_d)

        # The result is a tuple of outputs
        actual_output = result[0]

        # Due to the size, we check for equality using numpy
        actual_output_np = actual_output.numpy()
        expected_output_np = expected_output.numpy()

        self.assertTrue(
            np.array_equal(actual_output_np, expected_output_np),
            "Mixed operations output does not match expected value"
        )

    def test_operators_with_constants(self):
        """
        Test executing a subgraph that includes constants.
        """
        # Create input tensors
        input_data_a = tf.constant([0xFF00FF00], dtype=tf.uint32)
        input_data_b = tf.constant([0x00FF00FF], dtype=tf.uint32)

        tensor_a = Tensor(input_data_a, name="tensor_a")
        tensor_b = Tensor(input_data_b, name="tensor_b")

        # Reset subgraph for this test
        self.subgraph = SubGraph(name="ConstantsSubGraph")
        self.subgraph.register_input(tensor_a)
        self.subgraph.register_input(tensor_b)

        # Create a constant tensor
        constant_value = tf.constant([0xAAAAAAAA], dtype=tf.uint32)
        constant_tensor = Tensor(constant_value, name="constant_tensor")
        self.subgraph.add_tensor(constant_tensor)

        # Perform BITWISE_XOR between tensor_a and the constant
        xor_output = self.subgraph.bitwise_xor(tensor_a, constant_tensor, name="XorWithConstant")
        # Perform BITWISE_OR between xor_output and tensor_b
        or_output = self.subgraph.bitwise_or(xor_output, tensor_b, name="OrOp")
        # Perform BITWISE_NOT on the result
        not_output = self.subgraph.bitwise_not(or_output, name="NotOp")

        # Register final output
        self.subgraph.register_output(not_output)

        # Expected computations
        expected_xor = tf.bitwise.bitwise_xor(input_data_a, constant_value)
        expected_or = tf.bitwise.bitwise_or(expected_xor, input_data_b)
        expected_output = tf.bitwise.invert(expected_or)

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_data_a, input_data_b)

        # The result is a tuple of outputs
        actual_output = result[0]

        self.assertTrue(
            tf.reduce_all(tf.equal(actual_output, expected_output)),
            "Operations with constants output does not match expected value"
        )

    def test_sequential_operations(self):
        """
        Test executing a subgraph with sequential dependencies.
        """
        # Create input tensor
        input_data = tf.constant([0x12345678], dtype=tf.uint32)
        input_tensor = Tensor(input_data, name="input_tensor")

        # Reset subgraph for this test
        self.subgraph = SubGraph(name="SequentialOperationsSubGraph")
        self.subgraph.register_input(input_tensor)

        # Perform a sequence of NOT operations
        not_output1 = self.subgraph.bitwise_not(input_tensor, name="NotOp1")
        not_output2 = self.subgraph.bitwise_not(not_output1, name="NotOp2")
        not_output3 = self.subgraph.bitwise_not(not_output2, name="NotOp3")
        # Final output
        self.subgraph.register_output(not_output3)

        # Expected computations
        expected_output = tf.bitwise.invert(input_data)  # An odd number of NOTs results in bitwise inversion

        # Get the executable function
        func = self.subgraph.execute_function()

        # Execute the function
        result = func(input_data)

        # The result is a tuple of outputs
        actual_output = result[0]

        # Evaluate tensor values to numpy arrays for comparison
        actual_output_val = actual_output.numpy()
        expected_output_val = expected_output.numpy()

        # Assert equality
        self.assertTrue(
            (actual_output_val == expected_output_val).all(),
            "Sequential NOT operations output does not match expected value"
        )

    def test_circular_dependency(self):
        """
        Test that a circular dependency raises an error.
        """
        # Create input tensor
        input_data = tf.constant([0x12345678], dtype=tf.uint32)
        input_tensor = Tensor(input_data, name="input_tensor")

        # Reset subgraph for this test
        self.subgraph = SubGraph(name="CircularDependencySubGraph")
        self.subgraph.register_input(input_tensor)

        # Attempt to create a circular dependency
        with self.assertRaises(ValueError) as context:
            # Perform operation that depends on its own output
            self.subgraph.add_operator(
                opcode=IoOpCode.OpCode.BITWISE_NOT,
                input_tensors=[input_tensor],
                output_tensors=[input_tensor],  # Output overwrites input
                name="CircularOp"
            )
        self.assertIn("creates a circular dependency", str(context.exception))

    def test_invalid_operator(self):
        """
        Test that using an unsupported operator raises an error.
        """
        # Create input tensors
        input_data_a = tf.constant([0xFFFFFFFF], dtype=tf.uint32)
        input_data_b = tf.constant([0x00000000], dtype=tf.uint32)

        tensor_a = Tensor(input_data_a, name="tensor_a")
        tensor_b = Tensor(input_data_b, name="tensor_b")

        # Reset subgraph for this test
        self.subgraph = SubGraph(name="InvalidOperatorSubGraph")
        self.subgraph.register_input(tensor_a)
        self.subgraph.register_input(tensor_b)

        # Attempt to add an unsupported operation
        try:
            self.subgraph.add_operator(
                opcode=999,  # Invalid opcode
                input_tensors=[tensor_a, tensor_b],
                output_tensors=[Tensor(tf.constant([0], dtype=tf.uint32), name="output_tensor")],
                name="InvalidOp"
            )
            self.fail("Expected NotImplementedError due to invalid opcode")
        except NotImplementedError as e:
            self.assertIn("Operator with opcode 999 is not implemented", str(e))

    def test_save_load(self):
        """
        Test the save and load methods of SubGraph.
        """
        import tempfile
        import os

        # Set up the subgraph
        self.subgraph.register_input(self.tensor_a)
        self.subgraph.register_input(self.tensor_b)
        result_tensor = self.subgraph.bitwise_and(self.tensor_a, self.tensor_b, name="BitwiseAndOp")
        self.subgraph.register_output(result_tensor)

        # Create a temporary file path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            # Save the subgraph to the file
            self.subgraph.save(file_path)

            # Load the subgraph from the file
            loaded_subgraph = SubGraph.load(file_path)

            # Verify that the loaded subgraph matches the original
            self.assertEqual(loaded_subgraph.name.decode('utf-8'), self.subgraph.name, "Subgraph name should match after loading")
            self.assertEqual(len(loaded_subgraph.tensors), len(self.subgraph.tensors),
                             "Number of tensors should match after loading")
            self.assertEqual(len(loaded_subgraph.operators), len(self.subgraph.operators),
                             "Number of operators should match after loading")

            # Verify tensors and create a mapping
            tensor_mapping = {}
            for original_tensor, loaded_tensor in zip(self.subgraph.tensors, loaded_subgraph.tensors):
                self.assertEqual(original_tensor.name, loaded_tensor.name.decode('utf-8'), "Tensor names should match after loading")
                self.assertEqual(original_tensor.tf_tensor.dtype, loaded_tensor.tf_tensor.dtype,
                                 "Tensor dtypes should match after loading")
                self.assertEqual(original_tensor.tf_tensor.shape, loaded_tensor.tf_tensor.shape,
                                 "Tensor shapes should match after loading")
                self.assertTrue(
                    tf.reduce_all(tf.equal(original_tensor.tf_tensor, loaded_tensor.tf_tensor)),
                    "Tensor values should match after loading",
                )
                tensor_mapping[original_tensor] = loaded_tensor

            # Verify operators
            for original_op, loaded_op in zip(self.subgraph.operators, loaded_subgraph.operators):
                self.assertEqual(original_op.opcode, loaded_op.opcode, "Operator opcodes should match after loading")
                self.assertEqual(original_op.inputs, loaded_op.inputs, "Operator inputs should match after loading")
                self.assertEqual(original_op.outputs, loaded_op.outputs, "Operator outputs should match after loading")
                # Compare operator args if any
                self.assertEqual(original_op.args, loaded_op.args, "Operator arguments should match after loading")

            # Optionally, test execution to ensure functionality remains the same
            input_values = {
                self.tensor_a: tf.constant([1, 0, 1], dtype=tf.uint8),
                self.tensor_b: tf.constant([1, 1, 0], dtype=tf.uint8)
            }
            original_output = self.subgraph.execute(input_values)

            # Prepare input values for loaded subgraph with mapped tensors
            loaded_input_values = {
                tensor_mapping[self.tensor_a]: input_values[self.tensor_a],
                tensor_mapping[self.tensor_b]: input_values[self.tensor_b]
            }
            loaded_output = loaded_subgraph.execute(loaded_input_values)

            # Compare outputs using tensors from the respective subgraphs
            for orig_tensor, loaded_tensor in zip(self.subgraph.outputs, loaded_subgraph.outputs):
                self.assertTrue(
                    tf.reduce_all(tf.equal(original_output[orig_tensor], loaded_output[loaded_tensor])),
                    "Execution results should match after loading"
                )

        finally:
            # Clean up the temporary file
            os.remove(file_path)

if __name__ == '__main__':
    unittest.main()