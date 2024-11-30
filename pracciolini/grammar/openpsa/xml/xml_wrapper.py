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
        # begin by setting the tag
        element.tag = self.tag

        # then, set all the required attrs, defaulting to an empty string if None
        for key in self.req_attrs:
            value = self.__dict__.get(key, "")
            element.set(key, value)

        # next, for all the optional attrs, only write the keys if not None
        for key in self.attrs.difference(self.req_attrs):
            value = self.__dict__.get(key)
            if value is not None:
                element.set(key, value)

        # finally, do this recursively for all the children
        for child in self.children:
            element.append(child)
        return element

    def __str__(self) -> str:
        str_rep = [
            f"[{self.tag}",
        ]

        # then, set all the required attrs, defaulting to an empty string if None
        for key in self.req_attrs:
            value = self.__dict__.get(key, "")
            str_rep.append(f"{key}='{value}'")

        # next, for all the optional attrs, only write the keys if not None
        for key in self.attrs.difference(self.req_attrs):
            value = self.__dict__.get(key)
            if value is not None:
                str_rep.append(f"{key}='{value}'")

        str_rep.append("]")
        for child in self.children:
            str_rep.append(child)
        str_rep.append(f"[/{self.tag}]")
        return " ".join(str_rep)

    @classmethod
    def from_xml(cls: type('XMLWrapper'), root: lxml.etree.ElementTree):
        if root is None:
            raise lxml.etree.ParserError("Invalid XML element: root cannot be None")

        if root.tag != cls.tag:
            raise lxml.etree.ParserError(f"Parsed element is not a {cls.tag}")

        if len(cls.req_attrs.intersection(set(root.attrib.keys()))) < len(cls.req_attrs):
            raise lxml.etree.ParserError("Some required keys are missing from parsed element")

        for req_key in cls.req_attrs:
            req_value = root.get(req_key)
            if req_value is None or req_value == '':
                raise lxml.etree.ParserError("Some required keys are missing values from parsed element")

        return cls(*[el for el in root], **root.attrib)