from pracciolini.grammar.openpsa.xml.define_event import EventDefinition, NamedEvent
from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class RuleDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-rule",
                                 class_type=self)
        super().__init__(*args, **kwargs)


class RuleReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="rule",
                                 class_type=self)
        super().__init__(*args, **kwargs)


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


class NotExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="not",
                                 children=ExpressionMeta.permitted_tags,
                                 class_type=self)
        super().__init__(*args, **kwargs)
