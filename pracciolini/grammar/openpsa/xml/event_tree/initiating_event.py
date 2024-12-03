from pracciolini.grammar.openpsa.xml.define_event import EventDefinition
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class InitiatingEventDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-initiating-event",
                                 class_type=self,
                                 attrs={"event-tree"})
        super().__init__(*args, **kwargs)