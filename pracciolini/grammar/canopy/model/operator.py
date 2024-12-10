from typing import List, Optional, Dict

from pracciolini.grammar.canopy.io import (
    Operator as IoOperator,
    OperatorArgs as IoOperatorArgs,
    OpCode as IoOpCode,
    KofNArgs as IoKofNArgs,
    ReshapeArgs as IoReshapeArgs,
    MonteCarloExpectedValueOptions as IoMonteCarloExpectedValueOptions,
)
from pracciolini.grammar.canopy.model.tensor import Tensor


class Operator:
    """
    Represents an operator (operation) in the subgraph.

    Attributes:
        opcode (IoOpCode): The opcode of the operator.
        inputs (List[int]): Indices of input tensors in the subgraph tensors list.
        outputs (List[int]): Indices of output tensors in the subgraph tensors list.
        args (Optional[OperatorArgs]): The arguments of the operator.
        name (Optional[str]): Optional name of the operator.
    """

    def __init__(
        self,
        opcode: IoOpCode,
        inputs: List[int],
        outputs: List[int],
        args: Optional['OperatorArgs'] = None,
        name: Optional[str] = None,
    ):
        self.opcode = opcode
        self.inputs = inputs
        self.outputs = outputs
        self.args = args
        self.name = name

    def to_graph(self, builder) -> int:
        """
        Serializes the operator to FlatBuffers using the builder.

        Args:
            builder (flatbuffers.Builder): The FlatBuffers builder.

        Returns:
            int: The offset in the FlatBuffer where the Operator is stored.
        """
        # Serialize inputs
        IoOperator.OperatorStartInputsVector(builder, len(self.inputs))
        for idx in reversed(self.inputs):
            builder.PrependInt32(idx)
        inputs_vector = builder.EndVector(len(self.inputs))

        # Serialize outputs
        IoOperator.OperatorStartOutputsVector(builder, len(self.outputs))
        for idx in reversed(self.outputs):
            builder.PrependInt32(idx)
        outputs_vector = builder.EndVector(len(self.outputs))

        # Serialize args
        if self.args:
            args_offset = self.args.to_graph(builder)
            args_type = self.args.get_io_operator_args_type()
        else:
            args_offset = 0
            args_type = IoOperatorArgs.OperatorArgs.NONE

        # Serialize name
        if self.name:
            name_offset = builder.CreateString(self.name)
        else:
            name_offset = None

        # Build Operator object
        IoOperator.OperatorStart(builder)
        IoOperator.OperatorAddOpcode(builder, self.opcode)
        IoOperator.OperatorAddInputs(builder, inputs_vector)
        IoOperator.OperatorAddOutputs(builder, outputs_vector)
        IoOperator.OperatorAddArgsType(builder, args_type)
        if args_offset:
            IoOperator.OperatorAddArgs(builder, args_offset)
        if name_offset:
            IoOperator.OperatorAddName(builder, name_offset)
        operator_offset = IoOperator.OperatorEnd(builder)

        return operator_offset

    @classmethod
    def from_graph(
        cls, io_operator: IoOperator, tensor_map: Dict[int, Tensor]
    ) -> 'Operator':
        """
        Deserializes an operator from a FlatBuffer.

        Args:
            io_operator (IoOperator): The deserialized Operator.
            tensor_map (Dict[int, Tensor]): Mapping from tensor indices to Tensors.

        Returns:
            Operator: The Operator instance.
        """
        opcode = io_operator.Opcode()

        # Deserialize inputs
        inputs = [io_operator.Inputs(i) for i in range(io_operator.InputsLength())]

        # Deserialize outputs
        outputs = [io_operator.Outputs(i) for i in range(io_operator.OutputsLength())]

        # Deserialize args
        args_type = io_operator.ArgsType()
        args_data = io_operator.Args()
        if args_type != IoOperatorArgs.OperatorArgs.NONE:
            args = OperatorArgs.from_graph(args_data, args_type)
        else:
            args = None

        name = io_operator.Name()

        return cls(
            opcode=opcode,
            inputs=inputs,
            outputs=outputs,
            args=args,
            name=name,
        )


class OperatorArgs:
    """
    Base class for operator arguments.

    Subclasses should implement serialization and deserialization methods.
    """

    def to_graph(self, builder) -> int:
        raise NotImplementedError()

    @staticmethod
    def from_graph(data, args_type):
        if args_type == IoOperatorArgs.OperatorArgs.KofNArgs:
            kofn_args = IoKofNArgs.KofNArgs()
            kofn_args.Init(data.Bytes, data.Pos)
            atleast = kofn_args.Atleast()
            return KofNOperatorArgs(atleast)
        elif args_type == IoOperatorArgs.OperatorArgs.ReshapeArgs:
            reshape_args = IoReshapeArgs.ReshapeArgs()
            reshape_args.Init(data.Bytes, data.Pos)
            new_shape = [reshape_args.NewShape(i) for i in range(reshape_args.NewShapeLength())]
            return ReshapeOperatorArgs(new_shape)
        elif args_type == IoOperatorArgs.OperatorArgs.MonteCarloExpectedValueOptions:
            mc_args = IoMonteCarloExpectedValueOptions.MonteCarloExpectedValueOptions()
            mc_args.Init(data.Bytes, data.Pos)
            ci_low = mc_args.CiLow()
            ci_high = mc_args.CiHigh()
            return MonteCarloExpectedValueOperatorArgs(ci_low, ci_high)
        else:
            return None


class KofNOperatorArgs(OperatorArgs):
    def __init__(self, atleast: int = 0):
        self.atleast = atleast

    def to_graph(self, builder) -> int:
        IoKofNArgs.KofNArgsStart(builder)
        IoKofNArgs.KofNArgsAddAtleast(builder, self.atleast)
        return IoKofNArgs.KofNArgsEnd(builder)

    def get_io_operator_args_type(self):
        return IoOperatorArgs.OperatorArgs.KofNArgs


class ReshapeOperatorArgs(OperatorArgs):
    def __init__(self, new_shape: List[int]):
        self.new_shape = new_shape

    def to_graph(self, builder) -> int:
        IoReshapeArgs.ReshapeArgsStartNewShapeVector(builder, len(self.new_shape))
        for dim in reversed(self.new_shape):
            builder.PrependInt32(dim)
        new_shape_vector = builder.EndVector(len(self.new_shape))

        IoReshapeArgs.ReshapeArgsStart(builder)
        IoReshapeArgs.ReshapeArgsAddNewShape(builder, new_shape_vector)
        return IoReshapeArgs.ReshapeArgsEnd(builder)

    def get_io_operator_args_type(self):
        return IoOperatorArgs.OperatorArgs.ReshapeArgs


class MonteCarloExpectedValueOperatorArgs(OperatorArgs):
    def __init__(self, ci_low: float = 0.05, ci_high: float = 0.95):
        self.ci_low = ci_low
        self.ci_high = ci_high

    def to_graph(self, builder) -> int:
        IoMonteCarloExpectedValueOptions.MonteCarloExpectedValueOptionsStart(builder)
        IoMonteCarloExpectedValueOptions.MonteCarloExpectedValueOptionsAddCiLow(builder, self.ci_low)
        IoMonteCarloExpectedValueOptions.MonteCarloExpectedValueOptionsAddCiHigh(builder, self.ci_high)
        return IoMonteCarloExpectedValueOptions.MonteCarloExpectedValueOptionsEnd(builder)

    def get_io_operator_args_type(self):
        return IoOperatorArgs.OperatorArgs.MonteCarloExpectedValueOptions
