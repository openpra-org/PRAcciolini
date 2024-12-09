from __future__ import annotations

import flatbuffers

import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class OperatorArgs(object):
  NONE: int
  KofNArgs: int
  ReshapeArgs: int
  MonteCarloExpectedValueOptions: int

