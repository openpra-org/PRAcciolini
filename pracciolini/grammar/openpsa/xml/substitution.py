from pracciolini.grammar.openpsa.xml.define_event import EventDefinition
from pracciolini.grammar.openpsa.xml.expression.meta import ReferenceMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class SubstitutionSourceDefinition(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="source",
                                 class_type=self,
                                 children=ReferenceMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class SubstitutionTargetDefinition(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="target",
                                 class_type=self,
                                 children=ReferenceMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class SubstitutionDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-substitution",
                                 class_type=self,
                                 attrs={"type"},
                                 children={"hypothesis", "source", "target"})
        super().__init__(*args, **kwargs)

