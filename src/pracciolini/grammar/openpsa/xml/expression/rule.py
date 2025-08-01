from pracciolini.grammar.openpsa.xml.define_event import EventDefinition, NamedEvent
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class RuleDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-rule",
                                 class_type=self,
                                 children={"rule", "collect-expression"})
        super().__init__(*args, **kwargs)


class RuleReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="rule",
                                 class_type=self)
        super().__init__(*args, **kwargs)