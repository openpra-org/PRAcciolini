from typing import Set, Any, Tuple
import lxml


class XMLWrapper:
    tag: str
    attrs: Set[str]
    req_attrs: Set[str]
    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.children : Tuple[Any, ...] = args

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def to_xml(self) -> lxml.etree.Element:
        element = lxml.etree.Element(self.tag)
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
            raise lxml.etree.ParserError("Some required keys are missing from parsed element")

        return cls(*[el for el in root], **root.attrib)