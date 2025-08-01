from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.expression.param_dist import SystemMissionTimeDependent
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class PeriodicTestExpression(SystemMissionTimeDependent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="periodic-test", class_type=self, children={"float", "bool"}, req_children={"float"})
        super().__init__(*args, **kwargs)


class HypothesisTestExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="hypothesis", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)