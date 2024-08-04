from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Name, Label, Attributes, Role
from pracciolini.grammar.openpsa.xml.identifier import BooleanConstant, Expression, Units


class HouseEventDefinition(XMLSerializable):
    def __init__(self, name, role=None, label=None, attributes=None, boolean_constant=None):
        self.name = Name(name)
        self.role = Role(role) if role else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None
        self.boolean_constant = BooleanConstant(boolean_constant) if boolean_constant is not None else None

    def to_xml(self):
        element = etree.Element("define-house-event")
        element.append(self.name.to_xml())
        if self.role:
            element.append(self.role.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        if self.boolean_constant:
            element.append(self.boolean_constant.to_xml())
        return element


class BasicEventDefinition(XMLSerializable):
    def __init__(self, name, role=None, label=None, attributes=None, expression=None):
        self.name = Name(name)
        self.role = Role(role) if role else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None
        self.expression = Expression(expression) if expression else None

    def to_xml(self):
        element = etree.Element("define-basic-event")
        element.append(self.name.to_xml())
        if self.role:
            element.append(self.role.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())
        if self.expression:
            element.append(self.expression.to_xml())
        return element


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
