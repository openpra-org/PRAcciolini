from pracciolini.grammar.openpsa.xml.define_event import NamedEvent, EventDefinition
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class EventTreeReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="event-tree",
                                 class_type=self)
        super().__init__(*args, **kwargs)


class EventTreeDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-event-tree",
                                 class_type=self,
                                 children={"define-initiating-event", "define-functional-event", "define-sequence", "initial-state", "define-branch"})
        super().__init__(*args, **kwargs)