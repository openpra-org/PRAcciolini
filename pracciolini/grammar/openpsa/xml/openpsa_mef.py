from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Label, Name, Attributes


class OpsaMef(XMLSerializable):
    def __init__(self, name=None, label=None, attributes=None):
        self.name = Name(name) if name else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None

    def to_xml(self):
        opsa_mef_elem = etree.Element("opsa-mef")
        if self.name:
            opsa_mef_elem.append(self.name.to_xml())
        if self.label:
            opsa_mef_elem.append(self.label.to_xml())
        if self.attributes:
            opsa_mef_elem.append(self.attributes.to_xml())
        return opsa_mef_elem
