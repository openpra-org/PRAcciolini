from typing import Set
import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.xml_wrapper import XMLWrapper


class InitiatingEventDefinition(XMLWrapper):
    tag: str = "define-initiating-event"
    attrs: Set[str] = {'name', 'event-tree'}
    req_attrs: Set[str] = {'name'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(tag=InitiatingEventDefinition.tag, *args, **kwargs)

    def to_xml(self) -> etree.Element:
        return super().to_xml()

    @classmethod
    def from_xml(cls: type(XMLWrapper), root: lxml.etree.ElementTree):
        return super().from_xml(root)