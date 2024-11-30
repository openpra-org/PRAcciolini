from typing import Set, Tuple, Any
import lxml
from lxml import etree


class XMLWrapper:
    tag: str
    attrs: Set[str]
    req_attrs: Set[str]
    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.children : Tuple[Any, ...] = args

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def to_xml(self) -> etree.Element:
        element = etree.Element(self.tag)
        element.tag = self.tag
        for key in self.attrs:
            value = self.__dict__.get(key)
            element.set(key, value)
        for child in self.children:
            element.append(child)
        return element

    @classmethod
    def from_xml(cls: type('XMLWrapper'), root: lxml.etree.ElementTree):
        if root is None:
            raise lxml.etree.ParserError("Invalid XML element: root cannot be None")

        if root.tag != cls.tag:
            raise lxml.etree.ParserError(f"Parsed element is not a {cls.tag}")

        if len(cls.req_attrs.intersection(set(root.attrib.keys()))) < len(cls.req_attrs):
            raise lxml.etree.ParserError(f"Some required keys are missing from parsed element")

        return cls(*[el for el in root], **root.attrib)


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