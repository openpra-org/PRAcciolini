import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class SequenceDefinition(XMLSerializable):
    def __init__(self, name: str):
        self.name: str = name

    def to_xml(self):
        element = etree.Element("define-sequence")
        element.set("name", self.name)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'SequenceDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        # get the name
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("sequence definition does not contain a name")
        return cls(name=name)

    def __str__(self):
        return f"sequence-definition: {self.name}"

class SequenceReference(XMLSerializable):
    def __init__(self, name: str):
        self.name: str = name

    def to_xml(self):
        element = etree.Element("sequence")
        element.set("name", self.name)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'SequenceReference':
        if root is None:
            raise lxml.etree.ParserError(root)
        # get the name
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("sequence reference does not contain a name")
        return cls(name=name)

    def __str__(self):
        return f"{self.name}"