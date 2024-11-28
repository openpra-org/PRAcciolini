import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class FunctionalEventDefinition(XMLSerializable):
    def __init__(self, name: str):
        self.name: str = name

    def to_xml(self):
        element = etree.Element("define-functional-event")
        element.set("name", self.name)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'FunctionalEventDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("functional event definition does not contain a name")
        return cls(name=name)

    def __str__(self):
        str_rep = [
            f"functional-event-definition: {self.name}",
        ]
        return ", ".join(str_rep)