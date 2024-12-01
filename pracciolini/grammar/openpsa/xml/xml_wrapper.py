from typing import Set, Tuple
from lxml.etree import ElementTree, Element, ParserError

from pracciolini.grammar.openpsa.xml.info import XMLInfo
from pracciolini.grammar.openpsa.xml.model_data.model_data import ModelData



class XMLWrapper:

    def __init__(self, info: XMLInfo, *args, **kwargs) -> None:
        self.info: XMLInfo = info
        self.__dict__.update(kwargs)
        self.children : Tuple[type(XMLWrapper), ...] = args

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def to_xml(self) -> Element:
        element = Element(self.info.tag)
        # begin by setting the tag
        element.tag = self.info.tag

        # then, set all the required attrs, defaulting to an empty string if None
        for key in self.info.req_attrs:
            value = self.__dict__.get(key, "")
            element.set(key, value)

        # next, for all the optional attrs, only write the keys if not None
        for key in self.info.attrs.difference(self.info.req_attrs):
            value = self.__dict__.get(key)
            if value is not None:
                element.set(key, value)

        # finally, do this recursively for all the children
        for child in self.children:
            element.append(child.to_xml())
        return element

    def __str__(self) -> str:
        str_rep = [
            f"[{self.info.tag}",
        ]

        # then, set all the required attrs, defaulting to an empty string if None
        for key in self.info.req_attrs:
            value = self.__dict__.get(key, "")
            str_rep.append(f"{key}='{value}'")

        # next, for all the optional attrs, only write the keys if not None
        for key in self.info.attrs.difference(self.info.req_attrs):
            value = self.__dict__.get(key)
            if value is not None:
                str_rep.append(f"{key}='{value}'")

        str_rep.append("]")
        for child in self.children:
            str_rep.append(child)
        str_rep.append(f"[/{self.info.tag}]")
        return " ".join(str_rep)

    @staticmethod
    def parse_xml(root: ElementTree):
        if root is None:
            raise ParserError("cannot parse unknown type")
        tag = root.tag
        match tag:
            case InitiatingEventDefinition.info.tag:
                return InitiatingEventDefinition.from_xml(root)
            case ModelData.info.tag:
                return ModelData.from_xml(root)
            case _:
                raise ParserError(f"unknown tag type:{tag}")

    @classmethod
    def from_xml(cls: type('XMLWrapper'), root: ElementTree):
        if root is None:
            raise ParserError("Invalid XML element: root cannot be None")

        if root.tag != cls.info.tag:
            raise ParserError(f"Parsed element is not a {cls.info.tag}")

        if len(cls.info.req_attrs.intersection(set(root.attrib.keys()))) < len(cls.info.req_attrs):
            raise ParserError("Some required keys are missing from parsed element")

        for req_key in cls.info.req_attrs:
            req_value = root.get(req_key)
            if req_value is None or req_value == '':
                raise ParserError("Some required keys are missing values from parsed element")

        children = [XMLWrapper.parse_xml(child) for child in list(root)]

        return cls(*children, **root.attrib)


class InitiatingEventDefinition(XMLWrapper):
    info: XMLInfo = XMLInfo()
    info.tag = "define-initiating-event"
    info.attrs = {'name', 'event-tree'}
    info.req_attrs = {'name'}
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(info=InitiatingEventDefinition.info, *args, **kwargs)
