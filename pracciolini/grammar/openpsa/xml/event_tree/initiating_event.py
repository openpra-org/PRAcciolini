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

    def __str__(self) -> str:
        str_rep = [
            f"[{self.tag}",
        ]
        for key in self.attrs:
            value = self.__dict__.get(key)
            str_rep.append(f"{key}='{value}'")
        str_rep.append("]")
        for child in self.children:
            str_rep.append(child)
        str_rep.append(f"[/{self.tag}]")
        return " ".join(str_rep)