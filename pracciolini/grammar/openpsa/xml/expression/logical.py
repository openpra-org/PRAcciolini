from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class CardinalityExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="cardinality",
                                 attrs={"min", "max"},
                                 req_attrs={"min", "max"},
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class AtleastExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="atleast",
                                 attrs={"min"},
                                 req_attrs={"min"},
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class ImplyExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="imply",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class OrExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="or",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class AndExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="and",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class NandExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="nand",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class NorExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="nor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class XnorExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="xnor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class XorExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="xor",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class NotExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="not",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)


class IffExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="iff",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)
