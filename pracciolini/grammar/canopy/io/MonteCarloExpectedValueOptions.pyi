from __future__ import annotations

import flatbuffers

import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class MonteCarloExpectedValueOptions(object):
  @classmethod
  def GetRootAs(cls, buf: bytes, offset: int) -> MonteCarloExpectedValueOptions: ...
  @classmethod
  def GetRootAsMonteCarloExpectedValueOptions(cls, buf: bytes, offset: int) -> MonteCarloExpectedValueOptions: ...
  @classmethod
  def MonteCarloExpectedValueOptionsBufferHasIdentifier(cls, buf: bytes, offset: int, size_prefixed: bool) -> bool: ...
  def Init(self, buf: bytes, pos: int) -> None: ...
  def CiLow(self) -> float: ...
  def CiHigh(self) -> float: ...
def MonteCarloExpectedValueOptionsStart(builder: flatbuffers.Builder) -> None: ...
def Start(builder: flatbuffers.Builder) -> None: ...
def MonteCarloExpectedValueOptionsAddCiLow(builder: flatbuffers.Builder, ciLow: float) -> None: ...
def MonteCarloExpectedValueOptionsAddCiHigh(builder: flatbuffers.Builder, ciHigh: float) -> None: ...
def MonteCarloExpectedValueOptionsEnd(builder: flatbuffers.Builder) -> uoffset: ...
def End(builder: flatbuffers.Builder) -> uoffset: ...
