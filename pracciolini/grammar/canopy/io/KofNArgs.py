# automatically generated by the FlatBuffers compiler, do not modify

# namespace: io

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class KofNArgs(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = KofNArgs()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsKofNArgs(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def KofNArgsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x50\x59\x31", size_prefixed=size_prefixed)

    # KofNArgs
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # KofNArgs
    def Atleast(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def KofNArgsStart(builder: flatbuffers.Builder):
    builder.StartObject(1)

def Start(builder: flatbuffers.Builder):
    KofNArgsStart(builder)

def KofNArgsAddAtleast(builder: flatbuffers.Builder, atleast: int):
    builder.PrependUint32Slot(0, atleast, 0)

def AddAtleast(builder: flatbuffers.Builder, atleast: int):
    KofNArgsAddAtleast(builder, atleast)

def KofNArgsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return KofNArgsEnd(builder)