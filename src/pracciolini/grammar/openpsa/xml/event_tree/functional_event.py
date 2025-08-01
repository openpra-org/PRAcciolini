from pracciolini.grammar.openpsa.xml.define_event import EventDefinition
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class FunctionalEventDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-functional-event",
                                 class_type=self,
                                 children={"define-branch"})
        super().__init__(*args, **kwargs)
