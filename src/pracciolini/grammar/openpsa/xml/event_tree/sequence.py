from pracciolini.grammar.openpsa.xml.define_event import EventDefinition, NamedEvent
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class SequenceDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-sequence",
                                 class_type=self,
                                 children={"event-tree", "collect-expression", "rule"})
        super().__init__(*args, **kwargs)


class SequenceReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="sequence",
                                 class_type=self)
        super().__init__(*args, **kwargs)