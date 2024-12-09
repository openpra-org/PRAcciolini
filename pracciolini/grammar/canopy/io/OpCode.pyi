from __future__ import annotations

import flatbuffers

import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class OpCode(object):
  BITWISE_NOT: int
  BITWISE_AND: int
  BITWISE_OR: int
  BITWISE_XOR: int
  BITWISE_K_OF_N: int
  BITWISE_NAND: int
  BITWISE_NOR: int
  BITWISE_XNOR: int
  BITMASK_ZEROS: int
  BITMASK_ONES: int
  BITWISE_CONV_ND: int
  BITWISE_CONV_1D: int
  BITWISE_CONV_2D: int
  BITWISE_FFT_ND: int
  BITWISE_FFT_1D: int
  BITWISE_FFT_2D: int
  LOGICAL_NOT: int
  LOGICAL_AND: int
  LOGICAL_OR: int
  LOGICAL_XOR: int
  LOGICAL_NAND: int
  LOGICAL_NOR: int
  LOGICAL_XNOR: int
  RESHAPE: int
  MC_EXPECT_VAL: int
  MC_VAR_LOSS: int

