from typing import Optional, Set

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.fault_tree.gate import GateDefinition
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class FaultTreeDefinition(XMLSerializable):
    def __init__(self,
                 name:str,
                 gates: Optional[Set[GateDefinition]]=None
        ):
        self.name: str = name
        self.gates: Set[GateDefinition] = gates if gates is not None else set()

    def to_xml(self):
        element = etree.Element("define-fault-tree")
        element.set("name", self.name)
        for gate in self.gates:
            element.append(gate.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'FaultTreeDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("name is missing in define-fault-tree")

        # parse gate definition list
        gate_definitions_xml = root.findall("define-gate")
        gates: Set[GateDefinition] = set()
        for gate_definition_xml in gate_definitions_xml:
            gate_definition: GateDefinition = GateDefinition.from_xml(gate_definition_xml)
            gates.add(gate_definition)

        return cls(name=name, gates=gates)

    def __str__(self):
        return (f"\nfault-tree-definition: name: {self.name}"
                f"\n\tgates: {len(self.gates)}")