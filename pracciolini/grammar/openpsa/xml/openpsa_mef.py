from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Label, Name, Attributes


class OpsaMef(XMLSerializable):
    def __init__(self, name=None, label=None, attributes=None, children=None):
        self.name = Name(name) if name else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None
        self.children = children if children else []

    def to_xml(self):
        opsa_mef_element = etree.Element("opsa-mef")
        if self.name:
            opsa_mef_element.append(self.name.to_xml())
        if self.label:
            opsa_mef_element.append(self.label.to_xml())
        if self.attributes:
            opsa_mef_element.append(self.attributes.to_xml())
        # Handling multiple child elements
        for child in self.children:
            opsa_mef_element.append(child.to_xml())
        return opsa_mef_element
