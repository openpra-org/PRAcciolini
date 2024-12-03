from typing import Literal

from lxml.etree import Element, ParserError

from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class Label(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="label", class_type=self)
        super().__init__(*args, **kwargs)

    @property
    def text(self) -> str:
        return self["text"]

    @text.setter
    def text(self, value: str):
        self["text"] = value

    @text.deleter
    def text(self):
        self["text"] = ""

    @classmethod
    def validate(cls, instance: 'Label'):
        super().validate(instance)
        if instance.text is None:
            raise ParserError(f"{instance.info.tag} in class {instance.info.classname} should contain text")

    @classmethod
    def from_xml(cls, root: Element):
        if root is None:
            raise ParserError("Invalid XML element: root cannot be None")

        info: XMLInfo = XMLInfo.get(root.tag)

        if root.tag != info.tag:
            raise ParserError(f"Parsed element {root.tag} is not a {info.tag}")

        if root.text is None:
            raise ParserError(f"Parsed element {root.tag} does not contain any text")

        return cls(text=root.text, **root.attrib)


class Units:
    """
    Represents a unit of measurement.

    Args:
        unit: The unit of measurement. Must be one of "bool", "int", "float", "hours", "hours-1", "years", "years-1", "fit", or "demands".

    Raises:
        ValueError: If the `unit` is invalid.
    """

    def __init__(self, unit: Literal["bool", "int", "float", "hours", "hours-1", "years", "years-1", "fit", "demands"]) -> None:
        if unit not in ["bool", "int", "float", "hours", "hours-1", "years", "years-1", "fit", "demands"]:
            raise ValueError(f"Invalid unit: {unit}. Must be one of 'bool', 'int', 'float', 'hours', 'hours-1', 'years', 'years-1', 'fit', 'demands'.")
        self.unit: str = unit

    def to_xml(self) -> str:
        """
        Returns the unit as a string, ready to be used as an XML attribute value.

        Returns:
            The unit as a string.
        """
        return self.unit