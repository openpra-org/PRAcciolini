from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class ArithmeticAddExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="add", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArithmeticNegativeExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="neg", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArthmeticSubtractExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="sub", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArthmeticMultiplyExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="mul", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArthmeticDivideExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="div", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)
