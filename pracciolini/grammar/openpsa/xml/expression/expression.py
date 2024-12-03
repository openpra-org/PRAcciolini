from typing import Any, Set

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.fault_tree.event_reference import GateReference, BasicEventReference
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class Expression(XMLSerializable):
    def __init__(self, content):
        self.content = content

    def to_xml(self):
        return self.content.to_xml()

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> Any:
        if root is None:
            raise lxml.etree.ParserError(root)
        tag = root.tag
        if tag is None:
            raise lxml.etree.ParserError(root)

        match tag:
            case "not":
                return NotExpression.from_xml(root)
            case "and":
                return AndExpression.from_xml(root)
            case "or":
                return OrExpression.from_xml(root)
            case "atleast":
                return AtLeastExpression.from_xml(root)
            case "gate":
                return GateReference.from_xml(root)
            case "basic-event":
                return BasicEventReference.from_xml(root)
            case _:
                raise lxml.etree.ParserError(f"unknown tag type:{tag}")

    def __str__(self):
        str_rep = [
            "expression",
            f"\texpression: {self.content}",
        ]
        return "\n".join(str_rep)



class NotExpression:
    def __init__(self, expression: Expression):
        self.expression: Expression = expression  # ExpressionDefinition instance

    def to_xml(self):
        element = etree.Element("not")
        element.append(self.expression.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'NotExpression':
        if root is None:
            raise lxml.etree.ParserError(root)
        child = root[0]
        if child is None:
            raise lxml.etree.ParserError(root)
        child_expression: Expression = Expression.from_xml(child)
        return NotExpression(expression=child_expression)

    def __str__(self):
        return f"not({self.expression})"

class AndExpression:
    def __init__(self, expressions: Set[Expression]):
        self.expressions = expressions

    def to_xml(self):
        element = etree.Element("and")
        for expr in self.expressions:
            element.append(expr.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'AndExpression':
        if root is None or len(root) == 0:
            raise lxml.etree.ParserError(root)

        expressions: Set[Expression] = set()
        for child_xml in root:
            child: Expression = Expression.from_xml(child_xml)
            expressions.add(child)
        return cls(expressions=expressions)

    def __str__(self):
        expressions = ", ".join([str(expression) for expression in self.expressions])
        return f"and[{len(self.expressions)}]({expressions})"


class OrExpression:
    def __init__(self, expressions):
        self.expressions = expressions  # List[ExpressionDefinition]

    def to_xml(self):
        element = etree.Element("or")
        for expr in self.expressions:
            element.append(expr.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'OrExpression':
        if root is None or len(root) == 0:
            raise lxml.etree.ParserError(root)

        expressions: Set[Expression] = set()
        for child_xml in root:
            child: Expression = Expression.from_xml(child_xml)
            expressions.add(child)
        return cls(expressions=expressions)

    def __str__(self):
        expressions = ", ".join([str(expression) for expression in self.expressions])
        return f"or[{len(self.expressions)}]({expressions})"

class AtLeastExpression:
    def __init__(self, min_value, expressions):
        self.min_value = min_value  # int
        self.expressions = expressions  # List[ExpressionDefinition]

    def to_xml(self):
        element = etree.Element("atleast")
        element.set("min", str(self.min_value))
        for expr in self.expressions:
            element.append(expr.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'AtLeastExpression':
        if root is None or len(root) == 0:
            raise lxml.etree.ParserError(root)

        expressions: Set[Expression] = set()
        for child_xml in root:
            child: Expression = Expression.from_xml(child_xml)
            expressions.add(child)

        min_value = int(root.get("min"))

        if min_value is None:
            print("no minimum set for atleast gate, will set to 0")
            min_value = 0

        elif min_value < 0:
            print("minimum for atleast gate was set to <0, setting to 0")
            min_value = 0

        elif min_value > len(expressions):
            print("minimum for atleast gate is larger than number of children, will set to max")
            min_value = len(expressions)

        return cls(expressions=expressions, min_value=min_value)

    def __str__(self):
        expressions = ", ".join([str(expression) for expression in self.expressions])
        return f"atleast[min={self.min_value},{len(self.expressions)}]({expressions})"