from pracciolini.grammar.openpsa.xml.define_event import EventDefinition
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class ComponentDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-component",
                                 class_type=self)
        super().__init__(*args, **kwargs)
