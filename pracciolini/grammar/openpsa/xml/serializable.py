import inspect
from abc import ABC, abstractmethod
import logging
from typing import Tuple, Set, Dict, Optional, List, Any, Literal, Union
from importlib import import_module
import lxml
from lxml import etree
from lxml.etree import ParserError, Element

logger = logging.getLogger(__name__)

__TAG_MAP__: Dict[str, 'XMLInfo'] = {}
__CLASS_TAG_MAP__: Dict[str, List['XMLInfo']] = {}


class XMLInfo:

    def __init__(self,
                 class_type: Any,
                 attrs: Optional[Set[str]] = None,
                 req_attrs: Optional[Set[str]] = None,
                 children: Optional[Set[str]] = None,
                 req_children: Optional[Set[str]] = None,
                 tag: Optional[str] = None,
                 attrs_extra_constraints: Optional[Dict[str, Set[str]]] = None,
                 children_extra_constraints: Optional[Dict[str, Set[str]]] = None,
        ):
        self.classname = class_type.__class__.__name__
        self.module_path = inspect.getmodule(class_type).__name__
        XMLInfo._validate(self)
        self.attrs: Set[str] = attrs if attrs is not None else set()
        self.req_attrs: Set[str] = req_attrs if req_attrs is not None else set()
        self.children: Set[str] = children if children is not None else set()
        self.req_children: Set[str] = req_children if req_children is not None else set()
        self.attrs_extra_constraints: Optional[Dict[str, Set[str]]] = attrs_extra_constraints
        self.children_extra_constraints: Optional[Dict[str, Set[str]]] = children_extra_constraints

    @staticmethod
    def _validate(info: 'XMLInfo') -> None:
        required = [info.classname, info.module_path]
        for item in required:
            if item is None or item == "":
                raise ValueError("XMLInfo required fields cannot be empty")

    @staticmethod
    def register(class_type: Any) -> None:
        class_type()

    redundant_calls = 0
    @staticmethod
    def update_registry(tag: str, info: 'XMLInfo') -> None:
        ## first, check for existing tags
        if tag in __TAG_MAP__:
            XMLInfo.redundant_calls += 1
            print(f"redundant calls: {XMLInfo.redundant_calls}")
            return
        XMLInfo._validate(info)
        __TAG_MAP__[tag] = info

    @staticmethod
    def get(tag: str, alternative: Optional['XMLInfo'] = None) -> Optional['XMLInfo']:
        return __TAG_MAP__.get(tag, alternative)

    def __str__(self):
        str_rep = [self.classname, self.module_path, f"attrs: ['{",".join(self.attrs)}']",
                   f"req_attrs: ['{",".join(self.req_attrs)}']", f"children: ['{",".join(self.children)}']",
                   f"req_children: ['{",".join(self.req_children)}']",]
        return ", ".join(str_rep)


def import_class(classinfo: XMLInfo):
    try:
        module = import_module(classinfo.module_path)
        return getattr(module, classinfo.classname)
    except (ImportError, AttributeError) as e:
        raise ImportError(classinfo.classname, e)



class XMLSerializable(ABC):
    """
    An abstract base class for objects that can be serialized to XML.

    Subclasses must implement the `to_xml` method, which should return an
    `lxml.etree.Element` object representing the object's XML structure.
    """
    def __init__(self, *args, **kwargs) -> None:

        if not "tag" in kwargs:
            raise ValueError(f"Cannot map XML without tag. Attempted to construct with args: {args}, kwargs: {kwargs}")
        if not "info" in kwargs:
            raise ValueError(f"Cannot serialize XML without XMLInfo. Attempted to construct with args: {args}, kwargs: {kwargs}")

        # take the info out of kwargs
        info = kwargs.pop("info")
        # set all my properties, including 'tag'
        self.__dict__.update(kwargs)
        self.children : Tuple[type(XMLSerializable), ...] = args

        # double check to make sure tag exists
        if not hasattr(self, "tag"):
            raise ValueError(f"Received tag {self["tag"]} but unable to locate it {args}, kwargs: {kwargs}")

        # update the registry to point to the provided XMLInfo object for the given tag
        XMLInfo.update_registry(self["tag"], info)

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @classmethod
    def to_expr(cls, expr_type: Optional[Literal["DNF", "CNF"]] = "DNF"):
        pass

    @staticmethod
    @abstractmethod
    def set_tag(cls):
        pass

    # @classmethod
    # def to_xml(self) -> etree.Element:
    #     element = etree.Element(self.info.tag)
    #     # begin by setting the tag
    #     element.tag = self.info.tag
    #     element.text = self["text"] if self["text"] is not None else None
    #
    #     # then, set all the required attrs, defaulting to an empty string if None
    #     for key in self.info.req_attrs:
    #         value = self.__dict__.get(key, "")
    #         element.set(key, value)
    #
    #     # next, for all the optional attrs, only write the keys if not None
    #     for key in self.info.attrs.difference(self.info.req_attrs):
    #         value = self.__dict__.get(key)
    #         if value is not None:
    #             element.set(key, value)
    #
    #     # finally, do this recursively for all the children
    #     for child in self.children:
    #         element.append(child.to_xml())
    #
    #     return element

    def to_xml(self, tag: Optional[str] = None) -> etree.Element:
        tag_to_use = tag if tag is not None and isinstance(tag, str) and tag != "" else self["tag"]
        info: XMLInfo = XMLInfo.get(tag_to_use, None)
        if info is None:
            raise ValueError(f"Cannot build xml from unregistered tag {tag} for object {self}")

        element = etree.Element(tag_to_use)
        # begin by setting the tag and text
        element.tag = tag_to_use
        element.text = getattr(self, "text", None)

        # then, set all the required attrs, defaulting to an empty string if None
        for key in info.req_attrs:
            value = self.__dict__.get(key, "")
            element.set(key, value)

        # next, for all the optional attrs, only write the keys if not None
        for key in info.attrs.difference(info.req_attrs):
            value = self.__dict__.get(key, None)
            if value is not None:
                element.set(key, value)

        if hasattr(self, "children"):
            # finally, do this recursively for all the children
            for child in self.children:
                element.append(child.to_xml())

        return element

    # @classmethod
    # def my_method(cls):
    #     """Abstract class method that must be implemented by all subclasses."""
    #     pass

    def __str__(self) -> str:
        str_rep = [
            f"({self["tag"]}:",
        ]

        info: XMLInfo = XMLInfo.get(self["tag"], None)
        attrs: Set[str] = info.attrs if info is not None else set()
        req_attrs: Set[str] = info.req_attrs if info is not None else set()

        req_attrs_rep = ["req_attrs:["]
        # get all the required attrs, defaulting to an empty string if None
        for key in req_attrs:
            value = self.__dict__.get(key, "")
            req_attrs_rep.append(f"{key}='{value}'")
        req_attrs_rep.append("]")
        str_rep.append(" ".join(req_attrs_rep))

        # next, for all the optional attrs, only write the keys if not None
        attrs_rep = ["attrs:["]
        for key in (attrs - req_attrs):
            value = self.__dict__.get(key, None)
            if value is not None:
                attrs_rep.append(f"{key}='{value}'")
        attrs_rep.append("]")
        str_rep.append(" ".join(attrs_rep))

        # finally, collect all the other items
        for key in (self.__dict__.keys() - attrs - req_attrs - {'children'}):
            value = self.__dict__.get(key, None)
            if value is not None:
                str_rep.append(f"{key}='{value}'")

        # add the text object and close the starting tag
        str_rep.append(f"text={self["text"]})" if self["text"] is not None else ")")

        ## iterate over all children
        for child in self.children:
            str_rep.append(child)

        ## close the tag
        str_rep.append(f"(/{self["tag"]})")
        return " ".join(str_rep)

    @staticmethod
    def build(root: Element):
        if root is None:
            raise ParserError("Invalid XML element: root cannot be None")

        tag = root.tag
        if root.tag is None:
            raise ParserError("Parsed element does not have a tag")

        class_info = XMLInfo.get(tag)
        if class_info is None:
            raise ParserError(f"Failed to parse tag {tag}: Not defined in OpenPSA XML __TAG_MAP__")

        imported_class = import_class(class_info)
        if imported_class is None:
            raise ParserError(f"Unable to import class {class_info.classname} for tag {tag}")

        class_instance = imported_class.from_xml(root)
        if not isinstance(class_instance, imported_class):
            raise ParserError(f"parsed event {tag} was not instantiated as requested type {class_info.classname}. It was instantiated as {class_instance.__class__.__name__}.")

        return class_instance

    @classmethod
    def validate(cls, instance: 'XMLSerializable'):
        #if instance.info is None:
        #    raise ValueError("XMLSerializable can never have an empty info attribute")
        return instance

    @classmethod
    def from_xml(cls, root: Element):
        if root is None:
            raise ParserError("Invalid XML element: root cannot be None")

        info: XMLInfo = XMLInfo.get(root.tag)

        # if root.tag != info.tag:
        #     raise ParserError(f"Parsed element {root.tag} is not a {info.tag}")

        attributes = set(root.attrib.keys())
        required_attributes = info.req_attrs
        missing_attributes = required_attributes - attributes
        if len(missing_attributes) > 0:
            raise ParserError(f"required keys {missing_attributes} are missing from parsed element {root.tag}: {lxml.etree.tostring(root, pretty_print=True)}")

        for req_key in info.req_attrs:
            req_value = root.get(req_key)
            if req_value is None or req_value == '':
                raise ParserError(f"Some required keys are missing values from parsed element {root.tag}")

        for req_child in info.req_children:
            for found_child in root.findall(req_child):
                if found_child is None:
                    raise ParserError(f"Required child {req_child} is missing from parsed element {root.tag}")

        children = []
        # iterate over all children exactly once
        for child in root:
            if child is not None and child.tag in info.children:
                class_instance = XMLSerializable.build(child)
                children.append(class_instance)

        instance = cls(*children, **root.attrib)
        return cls.validate(instance)
