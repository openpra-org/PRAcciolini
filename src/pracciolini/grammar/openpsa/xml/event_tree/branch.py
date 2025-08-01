from pracciolini.grammar.openpsa.xml.define_event import EventDefinition, NamedEvent
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class BranchDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-branch",
                                 class_type=self,
                                 children={"fork", "branch", "sequence", "event-tree"})
        super().__init__(*args, **kwargs)


class BranchReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="branch",
                                 class_type=self)
        super().__init__(*args, **kwargs)
