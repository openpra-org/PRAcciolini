from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Label, Name, Attributes


class FaultTreeDefinition(XMLSerializable):
    def __init__(self, name, label=None, attributes=None, children=None):
        self.name = Name(name)
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None
        self.children = children if children else []

    def to_xml(self):
        element = etree.Element("define-fault-tree")
        element.append(self.name.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        for child in self.children:
            element.append(child.to_xml())
        return element
