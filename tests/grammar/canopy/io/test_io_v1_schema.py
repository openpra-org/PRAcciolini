import unittest

import flatbuffers

from pracciolini.grammar.canopy.io.Buffer import BufferStart, BufferEnd
from pracciolini.grammar.canopy.io.DAGs import DAGsStartBuffersVector, DAGsStartSubgraphsVector, DAGsStart, \
    DAGsAddSubgraphs, DAGsAddBuffers, DAGsAddName, DAGsAddDescription, DAGsEnd, DAGs
from pracciolini.grammar.canopy.io.OpCode import OpCode
from pracciolini.grammar.canopy.io.Operator import OperatorStartOutputsVector, OperatorStart, OperatorAddOpcode, \
    OperatorAddInputs, OperatorAddOutputs, OperatorAddName, OperatorEnd, OperatorStartInputsVector
from pracciolini.grammar.canopy.io.SubGraph import SubGraphStartTensorsVector, SubGraphStartOperatorsVector, \
    SubGraphStartInputsVector, SubGraphStartOutputsVector, SubGraphAddTensors, SubGraphStart, SubGraphAddInputs, \
    SubGraphAddOutputs, SubGraphAddOperators, SubGraphAddName, SubGraphEnd
from pracciolini.grammar.canopy.io.Tensor import TensorStartShapeVector, TensorStart, TensorAddShape, TensorAddName, \
    TensorEnd
from pracciolini.grammar.canopy.io.TensorType import TensorType


class TestIOSchemaV1(unittest.TestCase):
    def test_write_dags_flatbuffer(self):
        # Create a builder with an initial size (can be adjusted as needed)
        builder = flatbuffers.Builder(1024)

        # ----------------------------------
        # Step 1: Create Buffers
        # ----------------------------------

        # Buffers are required to have at least one empty buffer at index 0
        # Start Buffer
        BufferStart(builder)
        # Do not add data (empty buffer)
        buffer0_offset = BufferEnd(builder)

        buffers_offsets = [buffer0_offset]

        # Create Buffers vector
        num_buffers = len(buffers_offsets)
        DAGsStartBuffersVector(builder, num_buffers)
        for buf_offset in reversed(buffers_offsets):
            builder.PrependUOffsetTRelative(buf_offset)
        buffers_vector_offset = builder.EndVector(num_buffers)

        # ----------------------------------
        # Step 2: Create Tensors
        # ----------------------------------

        # Create first tensor (Tensor0)
        # Shape: [1], Type: default (UINT32), BufferIdx: 0 (referring to buffer0)
        shape0 = [1]
        TensorStartShapeVector(builder, len(shape0))
        for dim in reversed(shape0):
            builder.PrependInt32(dim)
        shape0_vector = builder.EndVector(len(shape0))

        tensor0_name_offset = builder.CreateString("Tensor0")

        TensorStart(builder)
        TensorAddShape(builder, shape0_vector)
        TensorAddName(builder, tensor0_name_offset)
        tensor0_offset = TensorEnd(builder)

        # Create second tensor (Tensor1)
        # Shape: [1], Type: default (UINT32), BufferIdx: 0 (referring to buffer0)
        shape1 = [1]
        TensorStartShapeVector(builder, len(shape1))
        for dim in reversed(shape1):
            builder.PrependInt32(dim)
        shape1_vector = builder.EndVector(len(shape1))

        tensor1_name_offset = builder.CreateString("Tensor1")

        TensorStart(builder)
        TensorAddShape(builder, shape1_vector)
        TensorAddName(builder, tensor1_name_offset)
        tensor1_offset = TensorEnd(builder)

        tensors_offsets = [tensor0_offset, tensor1_offset]

        # Create Tensors vector
        num_tensors = len(tensors_offsets)
        SubGraphStartTensorsVector(builder, num_tensors)
        for tensor_offset in reversed(tensors_offsets):
            builder.PrependUOffsetTRelative(tensor_offset)
        tensors_vector_offset = builder.EndVector(num_tensors)

        # ----------------------------------
        # Step 3: Create Operators
        # ----------------------------------

        # Create Operator (Operator0) with opcode BITWISE_NOT
        # Inputs: [0], Outputs: [1]
        inputs = [0]
        OperatorStartInputsVector(builder, len(inputs))
        for input_idx in reversed(inputs):
            builder.PrependInt32(input_idx)
        inputs_vector_offset = builder.EndVector(len(inputs))

        outputs = [1]
        OperatorStartOutputsVector(builder, len(outputs))
        for output_idx in reversed(outputs):
            builder.PrependInt32(output_idx)
        outputs_vector_offset = builder.EndVector(len(outputs))

        operator_name_offset = builder.CreateString("Operator0")

        OperatorStart(builder)
        OperatorAddOpcode(builder, OpCode.BITWISE_NOT)
        OperatorAddInputs(builder, inputs_vector_offset)
        OperatorAddOutputs(builder, outputs_vector_offset)
        OperatorAddName(builder, operator_name_offset)
        operator0_offset = OperatorEnd(builder)

        operators_offsets = [operator0_offset]

        # Create Operators vector
        num_operators = len(operators_offsets)
        SubGraphStartOperatorsVector(builder, num_operators)
        for op_offset in reversed(operators_offsets):
            builder.PrependUOffsetTRelative(op_offset)
        operators_vector_offset = builder.EndVector(num_operators)

        # ----------------------------------
        # Step 4: Create SubGraph
        # ----------------------------------

        # Inputs and Outputs for SubGraph
        SubGraphStartInputsVector(builder, len(inputs))
        for input_idx in reversed(inputs):
            builder.PrependInt32(input_idx)
        subgraph_inputs_vector_offset = builder.EndVector(len(inputs))

        SubGraphStartOutputsVector(builder, len(outputs))
        for output_idx in reversed(outputs):
            builder.PrependInt32(output_idx)
        subgraph_outputs_vector_offset = builder.EndVector(len(outputs))

        subgraph_name_offset = builder.CreateString("MainSubGraph")

        # Build SubGraph
        SubGraphStart(builder)
        SubGraphAddTensors(builder, tensors_vector_offset)
        SubGraphAddInputs(builder, subgraph_inputs_vector_offset)
        SubGraphAddOutputs(builder, subgraph_outputs_vector_offset)
        SubGraphAddOperators(builder, operators_vector_offset)
        SubGraphAddName(builder, subgraph_name_offset)
        subgraph0_offset = SubGraphEnd(builder)

        subgraphs_offsets = [subgraph0_offset]

        # Create SubGraphs vector
        num_subgraphs = len(subgraphs_offsets)
        DAGsStartSubgraphsVector(builder, num_subgraphs)
        for sg_offset in reversed(subgraphs_offsets):
            builder.PrependUOffsetTRelative(sg_offset)
        subgraphs_vector_offset = builder.EndVector(num_subgraphs)

        # ----------------------------------
        # Step 5: Create DAGs
        # ----------------------------------

        # Create name and description strings for DAGs
        name_offset = builder.CreateString("Test DAGs")
        description_offset = builder.CreateString("Test DAGs description")

        # Build DAGs
        DAGsStart(builder)
        DAGsAddSubgraphs(builder, subgraphs_vector_offset)
        DAGsAddBuffers(builder, buffers_vector_offset)
        DAGsAddName(builder, name_offset)
        DAGsAddDescription(builder, description_offset)
        dags_offset = DAGsEnd(builder)

        # Finish builder with the root object and file identifier
        builder.Finish(dags_offset, file_identifier=b"CPY1")

        # Get the byte array representing the flatbuffer
        buf = builder.Output()

        # Write the buffer to a binary file
        file_path = '/tmp/test_dags.cnpy'
        with open(file_path, 'wb') as f:
            f.write(buf)

        # ----------------------------------
        # Step 6: Verify the flatbuffer
        # ----------------------------------

        # Read buffer from file
        with open(file_path, 'rb') as f:
            read_buf = f.read()

        # Verify that read_buf matches buf
        self.assertEqual(buf, read_buf)

        # Parse the DAGs from the buffer
        read_dags = DAGs.GetRootAs(read_buf, 0)

        # Check that name and description match
        print(read_dags.Name())
        self.assertEqual(read_dags.Name().decode('utf-8'), "Test DAGs")
        self.assertEqual(read_dags.Description().decode('utf-8'), "Test DAGs description")

        # Check that buffers length is 1
        self.assertEqual(read_dags.BuffersLength(), 1)

        # Check that subgraphs length is 1
        self.assertEqual(read_dags.SubgraphsLength(), 1)

        # Get the subgraph
        subgraph = read_dags.Subgraphs(0)

        # Check that subgraph name matches
        self.assertEqual(subgraph.Name().decode('utf-8'), "MainSubGraph")

        # Check tensors length
        self.assertEqual(subgraph.TensorsLength(), 2)

        # Verify first tensor
        tensor0 = subgraph.Tensors(0)
        self.assertEqual(tensor0.Name().decode('utf-8'), "Tensor0")
        self.assertEqual(tensor0.ShapeLength(), 1)
        self.assertEqual(tensor0.Shape(0), 1)
        self.assertEqual(tensor0.Type(), TensorType.UINT32)
        self.assertEqual(tensor0.BufferIdx(), 0)

        # Verify second tensor
        tensor1 = subgraph.Tensors(1)
        self.assertEqual(tensor1.Name().decode('utf-8'), "Tensor1")
        self.assertEqual(tensor1.ShapeLength(), 1)
        self.assertEqual(tensor1.Shape(0), 1)
        self.assertEqual(tensor1.Type(), TensorType.UINT32)
        self.assertEqual(tensor1.BufferIdx(), 0)

        # Check Operators length
        self.assertEqual(subgraph.OperatorsLength(), 1)

        # Verify operator
        operator = subgraph.Operators(0)
        self.assertEqual(operator.Name().decode('utf-8'), "Operator0")
        self.assertEqual(operator.Opcode(), OpCode.BITWISE_NOT)
        self.assertEqual(operator.InputsLength(), 1)
        self.assertEqual(operator.Inputs(0), 0)
        self.assertEqual(operator.OutputsLength(), 1)
        self.assertEqual(operator.Outputs(0), 1)

        print("Flatbuffer created and verified successfully.")

if __name__ == '__main__':
    unittest.main()
