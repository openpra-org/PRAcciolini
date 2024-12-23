from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable

class LogicalExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def referenced_events(self):
        return self.children


class CardinalityExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="cardinality",
                                 attrs={"min", "max"},
                                 req_attrs={"min", "max"},
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class AtleastExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="atleast",
                                 attrs={"min"},
                                 req_attrs={"min"},
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class ImplyExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="imply",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class OrExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="or",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class AndExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="and",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class NandExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="nand",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class NorExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="nor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class XnorExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="xnor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class XorExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="xor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class NotExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="not",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class IffExpression(LogicalExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="iff",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)
