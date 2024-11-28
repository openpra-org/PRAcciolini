from typing import Optional

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.expression import Expression, FloatExpression
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Name, Label, Attributes, Role
from pracciolini.grammar.openpsa.xml.identifier import BooleanConstant, Units


class HouseEventDefinition(XMLSerializable):
    def __init__(self, name: str, boolean_constant: Optional[BooleanConstant] = None):
        self.name: str = name
        self.boolean_constant: Optional[BooleanConstant] = boolean_constant if boolean_constant is not None else None

    def to_xml(self):
        element = etree.Element("define-house-event")
        element.set("name", self.name)
        if self.boolean_constant:
            element.append(self.boolean_constant.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'HouseEventDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("house-event definition does not contain a name")
        constant_xml = root.find("constant")
        constant: Optional[BooleanConstant] = None
        if constant_xml is not None:
            constant = BooleanConstant.from_xml(constant_xml)
        return cls(name=name, boolean_constant=constant)

    def __str__(self):
        return f"house-event-definition: name:{self.name}, constant:{self.boolean_constant.__str__()}"

class BasicEventDefinition(XMLSerializable):
    def __init__(self, name, role=None, label=None, attributes=None, expression=None):
        self.name: str = name
        self.role = Role(role) if role else None
        self.label = Label(label) if label else None
        self.value = FloatExpression(expression) if expression else None
        self.attributes = Attributes(attributes) if attributes else None
        self.expression = Expression(expression) if expression else None

    def to_xml(self):
        element = etree.Element("define-basic-event")
        element.append(self.name)
        if self.role:
            element.append(self.role.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        if self.expression:
            element.append(self.expression.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'BasicEventDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        label: Label = Label.from_xml(root.find("label"))
        expr: FloatExpression = FloatExpression.from_xml(root.find("float"))
        basic_event: BasicEventDefinition = BasicEventDefinition(name=name, label=label, expression=expr)
        return basic_event

    def __str__(self):
        return f"basic-event-definition: name:{self.name}, value:{self.value.__str__()}, label:{self.label}"

class ParameterDefinition(XMLSerializable):
    def __init__(self, name, role=None, unit=None, label=None, attributes=None, expression=None):
        self.name = Name(name)
        self.role = Role(role) if role else None
        self.unit = Units(unit) if unit else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None
        self.expression = Expression(expression)

    def to_xml(self):
        element = etree.Element("define-parameter")
        element.append(self.name.to_xml())
        if self.role:
            element.append(self.role.to_xml())
        if self.unit:
            element.set("unit", self.unit.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        element.append(self.expression.to_xml())
        return element
