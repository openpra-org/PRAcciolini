from __future__ import annotations

import flatbuffers

import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class TensorType(object):
  UINT32: int
  UINT4: int
  UINT8: int
  UINT16: int
  UINT64: int
  FLOAT16: int
  FLOAT32: int
  FLOAT64: int

