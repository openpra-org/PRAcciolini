from lxml import etree
from typing import Literal


class XMLSerializable:
    def to_xml(self):
        raise NotImplementedError("Subclasses must implement this method.")


class Identifier(XMLSerializable):
    def __init__(self, identifier):
        self.identifier = identifier

    def to_xml(self):
        return self.identifier


class NonEmptyString(XMLSerializable):
    def __init__(self, text):
        self.text = text

    def to_xml(self):
        return self.text


class Label(XMLSerializable):
    def __init__(self, text):
        self.text = NonEmptyString(text)

    def to_xml(self):
        label_elem = etree.Element("label")
        label_elem.text = self.text.to_xml()
        return label_elem


class Name(XMLSerializable):
    def __init__(self, identifier):
        self.identifier = Identifier(identifier)

    def to_xml(self):
        name_elem = etree.Element("name")
        name_elem.text = self.identifier.to_xml()
        return name_elem


class Attribute(XMLSerializable):
    def __init__(self, name, value, type=None):
        self.name = Name(name)
        self.value = NonEmptyString(value)
        self.type = NonEmptyString(type) if type else None

    def to_xml(self):
        attribute_elem = etree.Element("attribute")
        attribute_elem.set("name", self.name.to_xml())
        attribute_elem.set("value", self.value.to_xml())
        if self.type:
            attribute_elem.set("type", self.type.to_xml())
        return attribute_elem


class Attributes(XMLSerializable):
    def __init__(self, attributes):
        self.attributes = [Attribute(**attr) for attr in attributes]

    def to_xml(self):
        attributes_elem = etree.Element("attributes")
        for attribute in self.attributes:
            attributes_elem.append(attribute.to_xml())
        return attributes_elem


class Role(XMLSerializable):
    def __init__(self, role_value):
        if role_value not in ["private", "public"]:
            raise ValueError("Invalid role value. Must be 'private' or 'public'.")
        self.role_value = role_value

    def to_xml(self):
        return self.role_value


class BooleanConstant(XMLSerializable):
    def __init__(self, value):
        if not isinstance(value, bool):
            raise ValueError("BooleanConstant value must be a boolean")
        self.value = value

    def to_xml(self):
        constant_elem = etree.Element("constant")
        constant_elem.set("value", str(self.value).lower())  # XML boolean values are typically "true" or "false"
        return constant_elem


class Expression(XMLSerializable):
    def __init__(self, content):
        self.content = content  # This could be a simple string or a structured object for complex expressions

    def to_xml(self):
        # For simplicity, assume content is just a string for now
        expression_elem = etree.Element("expression")
        expression_elem.text = self.content
        return expression_elem


class Units:
    def __init__(self, unit: Literal["bool", "int", "float", "hours", "hours-1", "years", "years-1", "fit", "demands"]):
        if unit not in ["bool", "int", "float", "hours", "hours-1", "years", "years-1", "fit", "demands"]:
            raise ValueError(f"Invalid unit: {unit}. Must be one of 'bool', 'int', 'float', 'hours', 'hours-1', "
                             f"'years', 'years-1', 'fit', 'demands'.")
        self.unit = unit

    def to_xml(self):
        # Since the unit is a simple attribute value, we return it directly.
        # The actual element attribute assignment will be handled by the parent element.
        return self.unit
