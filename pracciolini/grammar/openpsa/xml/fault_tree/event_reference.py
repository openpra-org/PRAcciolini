from pracciolini.grammar.openpsa.xml.define_event import NamedEvent
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class GateReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="gate",
                                 class_type=self)
        super().__init__(*args, **kwargs)


class BasicEventReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="basic-event",
                                 class_type=self)
        super().__init__(*args, **kwargs)
