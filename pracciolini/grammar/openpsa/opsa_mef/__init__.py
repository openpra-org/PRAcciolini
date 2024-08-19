from lxml import etree

class OpsaMef:
    def __init__(self, xml_root):
        self.name = None
        self.label = None
        self.attributes = None
        self.elements = []

        if not xml_root.tag == "opsa-mef":
            raise ValueError("Invalid XML root element, expected 'opsa-mef'")

        # Optional name
        name_elem = xml_root.find(".//name")
        if name_elem is not None:
            self.name = Name.from_xml(name_elem)

        # Optional label
        label_elem = xml_root.find(".//label")
        if label_elem is not None:
            self.label = Label.from_xml(label_elem)

        # Optional attributes
        attributes_elem = xml_root.find(".//attributes")
        if attributes_elem is not None:
            self.attributes = Attributes.from_xml(attributes_elem)

        # Handling multiple possible child elements
        for child in xml_root:
            if child.tag == "define-fault-tree":
                self.elements.append(FaultTreeDefinition.from_xml(child))
            elif child.tag == "model-data":
                self.elements.append(ModelData.from_xml(child))
            # Add other element types similarly
            # elif child.tag == "event-tree-definition":
            #     self.elements.append(EventTreeDefinition.from_xml(child))
            # elif child.tag == "alignment-definition":
            #     self.elements.append(AlignmentDefinition.from_xml(child))
            # etc.

    def to_xml(self):
        root = etree.Element("opsa-mef")

        if self.name:
            root.append(self.name.to_xml())
        if self.label:
            root.append(self.label.to_xml())
        if self.attributes:
            root.append(self.attributes.to_xml())

        for element in self.elements:
            root.append(element.to_xml())

        return root

    @staticmethod
    def from_xml(xml_root):
        return OpsaMef(xml_root)