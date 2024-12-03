from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class CollectExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="collect-expression",
                                 class_type=self,
                                 children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class CollectFormula(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="collect-formula",
                                 class_type=self,
                                 children=ExpressionMeta.permitted_tags.union({"basic-event"})
                                 )
        super().__init__(*args, **kwargs)
