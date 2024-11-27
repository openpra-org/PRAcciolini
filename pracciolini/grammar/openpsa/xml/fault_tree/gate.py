from typing import Optional

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.expression import Expression
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class GateDefinition(XMLSerializable):
    def __init__(self, name: str, expression: Optional[Expression], role: Optional[str]=None):
        self.name: str = name
        self.expression = expression
        self.role: Optional[str] = role

    def to_xml(self):
        element = etree.Element("define-gate")
        element.set("name", self.name)
        if self.role:
            element.set("role", self.role)
        element.append(self.expression.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'GateDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)

        # parse the subexpression first
        content = root[0]
        if content is None:
            raise lxml.etree.ParserError("gate with undefined content")
        expression: Expression = Expression.from_xml(root=content)

        # parse the name
        dot_path = root.get("name")
        if dot_path is None:
            raise lxml.etree.ParserError("attr name is missing in gate definition")

        # parse the role string
        role = root.get("role")

        return cls(name=dot_path, expression=expression, role=role)

    def __str__(self):
        return (f"\ngate-definition: name: {self.name}"
                f"\trole: {self.role}"
                f"\n\texpression: {self.expression}")
