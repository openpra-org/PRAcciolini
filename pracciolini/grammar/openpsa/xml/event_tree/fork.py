from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class ForkDefinition(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="fork",
                                 class_type=self,
                                 attrs={"functional-event"},
                                 req_attrs={"functional-event"},
                                 children={"path"},
                                 req_children={"path"})
        super().__init__(*args, **kwargs)


class PathDefinition(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="path",
                                 class_type=self,
                                 attrs={"state"},
                                 req_attrs={"state"},
                                 children={"collect-formula", "collect-expression", "fork", "sequence", "event-tree", "branch"},
                                 req_children={"collect-formula"})
        super().__init__(*args, **kwargs)
