from pracciolini.grammar.openpsa.xml.define_event import EventDefinition, NamedEvent
from pracciolini.grammar.openpsa.xml.expression.meta import LogicalMeta, ReferenceMeta, ConstantsMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo


class FaultTreeDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-fault-tree",
                                 class_type=self,
                                 children={"define-gate", "define-house-event", "define-basic-event", "define-parameter"})
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'FaultTreeDefinition'):
        super().validate(instance)
        return instance


class GateDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-gate",
                                 class_type=self,
                                 children=LogicalMeta.permitted_tags.union(ReferenceMeta.permitted_tags).union(ConstantsMeta.permitted_tags))
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'GateDefinition'):
        super().validate(instance)
        return instance


class GateReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="gate",
                                 class_type=self)
        super().__init__(*args, **kwargs)