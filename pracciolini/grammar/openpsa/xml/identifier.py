import lxml
from lxml import etree
from typing import Literal, Optional, List, Any


class XMLSerializableAttribute:
    def __init__(self, key: str, value: Any):
        self.key: str = key
        self.value: Any = value

    def to_xml(self, element: etree.Element) -> etree.Element:
        element.set(self.key, self.value)
        return element


class XMLSerializable:
    """
    An abstract base class for objects that can be serialized to XML.

    Subclasses must implement the `to_xml` method, which should return an
    `lxml.etree.Element` object representing the object's XML structure.
    """

    def to_xml(self) -> etree.Element:
        """
        Serializes the object to an XML element.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            An `lxml.etree.Element` object representing the XML structure.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Identifier(XMLSerializable):
    """
    Represents an identifier.

    Args:
        identifier: The identifier value.
    """

    def __init__(self, identifier: str) -> None:
        self.identifier: str = identifier

    def to_xml(self) -> etree.Element:
        """
        Serializes the identifier to an XML element.

        Returns:
            An `lxml.etree.Element` object containing the identifier text.
        """
        return etree.Element("identifier", text=self.identifier)


class NonEmptyString(XMLSerializable):
    """
    Represents a non-empty string.

    Args:
        text: The string value.
    """

    def __init__(self, text: str) -> None:
        self.text: str = str(text)

    def to_xml(self) -> etree.Element:
        """
        Serializes the string to an XML element.

        Returns:
            An `lxml.etree.Element` object containing the string text.
        """
        return etree.Element("text", text=self.text)

    def __str__(self) -> str:
        """
        Returns the string representation of the object.

        Returns:
            The string value.
        """
        return self.text


class Label(XMLSerializable):
    """
    Represents a label.

    Args:
        text: The label text.
    """

    def __init__(self, text: str) -> None:
        self.text: NonEmptyString = NonEmptyString(text)

    def to_xml(self) -> etree.Element:
        """
        Serializes the label to an XML element.

        Returns:
            An `lxml.etree.Element` object representing the label XML structure.
        """
        label_elem = etree.Element("label")
        label_elem.append(self.text.to_xml())
        return label_elem

    @classmethod
    def from_xml(cls, root: Optional[etree.Element]) -> 'Label':
        """
        Deserializes an XML element into a Label object.

        Args:
            root: The root XML element representing the label.

        Raises:
            lxml.etree.ParserError: If the root element is None or invalid.

        Returns:
            A Label object representing the deserialized label.
        """
        if root is None:
            raise lxml.etree.ParserError(root)
        return Label(text=root.text)

    def __str__(self) -> str:
        """
        Returns the string representation of the label.

        Returns:
            The label text.
        """
        return str(self.text)

class Name(XMLSerializableAttribute):
    def __init__(self, value: str) -> None:
        super().__init__(key="name", value=value)


class Attribute(XMLSerializable):
    """
    Represents an attribute within an XML structure.

    Args:
        name: The name of the attribute.
        value: The value of the attribute.
        type (Optional[str]): The optional type of the attribute (default: None).
    """

    def __init__(self, name: str, value: str, type: Optional[str] = None) -> None:
        self.name: Name = Name(name)  # Type hint for clarity
        self.value: NonEmptyString = NonEmptyString(value)
        self.type: Optional[NonEmptyString] = NonEmptyString(type) if type else None

    def to_xml(self) -> etree.Element:
        """
        Serializes the attribute to an XML element.

        Returns:
            An `lxml.etree.Element` object representing the attribute XML structure.
        """
        attribute_elem = etree.Element("attribute")
        attribute_elem.set("name", self.name.to_xml())
        attribute_elem.set("value", self.value.to_xml())
        if self.type:
            attribute_elem.set("type", self.type.to_xml())
        return attribute_elem


class Attributes(XMLSerializable):
    """
    Represents a collection of attributes within an XML structure.

    Args:
        attributes: A list of dictionaries, where each dictionary represents an attribute
                    with keys "name", "value", and optional "type".
    """

    def __init__(self, attributes: List[dict]) -> None:
        self.attributes: List[Attribute] = [Attribute(**attr) for attr in attributes]

    def to_xml(self) -> etree.Element:
        """
        Serializes the collection of attributes to an XML element.

        Returns:
            An `lxml.etree.Element` object representing the attributes XML structure.
        """
        attributes_elem = etree.Element("attributes")
        for attribute in self.attributes:
            attributes_elem.append(attribute.to_xml())
        return attributes_elem


class Role(XMLSerializableAttribute):
    """
    Represents a role, which can be either "private" or "public".

    Args:
        value: The role value, either "private" or "public".

    Raises:
        ValueError: If the `value` is invalid.
    """
    def __init__(self, value: Literal["private", "public"]) -> None:
        if value not in ["private", "public"]:
            raise ValueError("Invalid role value. Must be 'private' or 'public'.")
        super().__init__(key="name", value=value)


class BooleanConstant(XMLSerializable):
    """
    Represents a boolean constant.

    Args:
        value: The boolean value of the constant.

    Raises:
        ValueError: If the `value` is not a boolean.
    """

    def __init__(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("BooleanConstant value must be a boolean")
        self.value: bool = value

    def to_xml(self) -> etree.Element:
        """
        Serializes the boolean constant to an XML element.

        Returns:
            An `lxml.etree.Element` object representing the boolean constant XML structure.
        """
        constant_elem = etree.Element("constant")
        constant_elem.set("value", str(self.value).lower())  # XML boolean values are typically "true" or "false"
        return constant_elem


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