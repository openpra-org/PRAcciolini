import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Role, Label, Attributes


class SequenceDefinition(XMLSerializable):
    def __init__(self, name, role=None, label=None, attributes=None):
        self.name: str = name
        self.role = Role(role) if role else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None

    def to_xml(self):
        element = etree.Element("define-sequence")
        element.set("name", self.name)
        if self.role:
            element.set("role", self.role.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'SequenceDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        sequence: SequenceDefinition = SequenceDefinition(name=name)
        return sequence

    def __str__(self):
        str_rep = [
            f"sequence-definition: {self.name}",
        ]
        if self.label is not None:
            str_rep.append(f"label:{self.label}")
        if self.role is not None:
            str_rep.append(f"role:{self.role}")
        return ", ".join(str_rep)

class SequenceReference(XMLSerializable):
    def __init__(self, name):
        self.name: str = name

    def to_xml(self):
        element = etree.Element("sequence")
        element.set("name", self.name)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'SequenceReference':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        sequence_ref: SequenceReference = SequenceReference(name=name)
        return sequence_ref

    def __str__(self):
        return f"{self.name}"