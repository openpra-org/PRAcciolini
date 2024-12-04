from pracciolini.grammar.openpsa.mx.factory import build_from_xml
from pracciolini.grammar.openpsa.mx.serializable import XMLSerializable
from pracciolini.grammar.openpsa.mx.decorators import xml_register
from lxml.etree import Element

@xml_register('and', 'or', 'not', 'xor', 'nor')
class LogicalExpression(XMLSerializable):
    def __init__(self, tag: str, children=None, **kwargs):
        self.tag = tag
        self.children = children if children is not None else []
        # Initialize other attributes

    @classmethod
    def from_xml(cls, element: Element):
        tag = element.tag
        children = [build_from_xml(child) for child in element]
        return cls(tag, children, **element.attrib)

    def to_xml(self) -> Element:
        element = Element(self.tag)
        # Set attributes
        for child in self.children:
            element.append(child.to_xml())
        return element