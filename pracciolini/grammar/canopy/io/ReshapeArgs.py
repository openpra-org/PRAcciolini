# automatically generated by the FlatBuffers compiler, do not modify

# namespace: io

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReshapeArgs(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReshapeArgs()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReshapeArgs(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ReshapeArgsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x50\x59\x31", size_prefixed=size_prefixed)

    # ReshapeArgs
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReshapeArgs
    def NewShape(self, j: int):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ReshapeArgs
    def NewShapeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ReshapeArgs
    def NewShapeLength(self) -> int:
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ReshapeArgs
    def NewShapeIsNone(self) -> bool:
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def ReshapeArgsStart(builder: flatbuffers.Builder):
    builder.StartObject(1)

def Start(builder: flatbuffers.Builder):
    ReshapeArgsStart(builder)

def ReshapeArgsAddNewShape(builder: flatbuffers.Builder, newShape: int):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(newShape), 0)

def AddNewShape(builder: flatbuffers.Builder, newShape: int):
    ReshapeArgsAddNewShape(builder, newShape)

def ReshapeArgsStartNewShapeVector(builder, numElems: int) -> int:
    return builder.StartVector(4, numElems, 4)

def StartNewShapeVector(builder, numElems: int) -> int:
    return ReshapeArgsStartNewShapeVector(builder, numElems)

def ReshapeArgsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return ReshapeArgsEnd(builder)