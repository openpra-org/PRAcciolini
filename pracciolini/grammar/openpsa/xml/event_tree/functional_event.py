import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Role, Label, Attributes


class FunctionalEventDefinition(XMLSerializable):
    def __init__(self, name, role=None, label=None, attributes=None):
        self.name: str = name
        self.role = Role(role) if role else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None

    def to_xml(self):
        element = etree.Element("define-functional-event")
        element.set("name", self.name)
        if self.role:
            element.set("role", self.role.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'FunctionalEventDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError(root)
        functional_event: FunctionalEventDefinition = FunctionalEventDefinition(name=name)
        return functional_event

    def __str__(self):
        str_rep = [
            f"functional-event-definition: {self.name}",
        ]
        if self.label is not None:
            str_rep.append(f"label:{self.label}")
        if self.role is not None:
            str_rep.append(f"role:{self.role}")
        return ", ".join(str_rep)