from abc import abstractmethod

from coverage.debug import simplify
from pyeda.inter import *

from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.reference import HouseEventReference, BasicEventReference
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable

class LogicalExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def symbol(self) -> str:
        pass

    def to_expr(self, simplify=False) -> str:
        # operands: List[str] = []
        # for child in self.children:
        #     if isinstance(BasicEventReference, child) or isinstance(HouseEventReference, child):
        #         operands.append(child.name)
        #     elif
        return f" {self.symbol()} ".join(child.name for child in self.children)

        #return self.symbol()


    def referenced_gates(self):



class CardinalityExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="cardinality",
                                 attrs={"min", "max"},
                                 req_attrs={"min", "max"},
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def symbol(self) -> str:
        return "?"


class AtleastExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="atleast",
                                 attrs={"min"},
                                 req_attrs={"min"},
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    # def to_expr(self, simplify=False) -> Expression:
    #     expr_vars = exprvar(tuple(child.name for child in self.children))
    #     op = Mux(*expr_vars.names, simplify=simplify)
    #     return op

    def symbol(self) -> str:
        return "?"

class ImplyExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="imply",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Implies(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "=>"

class OrExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="or",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Or(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "|"

class AndExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="and",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = And(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "&"

class NandExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="nand",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Nand(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "?"

class NorExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="nor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Nor(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "?"


class XnorExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="xnor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Xnor(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "?"

class XorExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="xor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Xor(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "^"


class NotExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="not",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)

    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Not(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "~"

class IffExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="iff",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


    def to_expr(self, simplify=False) -> Expression:
        expr_vars = exprvar(tuple(child.name for child in self.children))
        op = Equal(*expr_vars.names, simplify=simplify)
        return op

    def symbol(self) -> str:
        return "?"