import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.expression import Expression
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class CollectFormulaDefinition(XMLSerializable):
    def __init__(self, expression: Expression):
        self.expression = expression  # ExpressionDefinition instance

    def to_xml(self):
        element = etree.Element("collect-formula")
        element.append(self.expression.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'CollectFormulaDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        if len(root) > 1:
            raise lxml.etree.ParserError("collect-formula should not have more than one descendant")
        expression_xml = root[0]
        expression: Expression = Expression.from_xml(expression_xml)
        collect_formula: CollectFormulaDefinition = CollectFormulaDefinition(expression)
        return collect_formula

    def __str__(self):
        return f"collect-formula({self.expression})"