from typing import List

from pracciolini.grammar.openpsa.xml.define_event import EventDefinition, NamedEvent
from pracciolini.grammar.openpsa.xml.expression.logical import LogicalExpression
from pracciolini.grammar.openpsa.xml.expression.meta import LogicalMeta, ReferenceMeta, ConstantsMeta
from pracciolini.grammar.openpsa.xml.reference import BasicEventReference, HouseEventReference
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

    @property
    def gates(self):
        gates: List[GateDefinition] = []
        for child in self.children:
            if isinstance(child, GateDefinition):
                gates.append(child)
        return gates

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

    @property
    def type(self):
        for child in self.children:
            if isinstance(child, LogicalExpression):
                return child.info.tag
        return None

    @property
    def referenced_events(self):
        if isinstance(self.children[0], LogicalExpression) and self.children[0].info.tag == "not":
            return self.children[0].referenced_events.referenced_events
        return self.children[0].referenced_events

    @property
    def referenced_events_by_type(self):
        basic_events = []
        gates = []
        for event_ref in self.referenced_events:
            if isinstance(event_ref, BasicEventReference):
                basic_events.append(event_ref)
            elif isinstance(event_ref, GateReference):
                gates.append(event_ref)
        return basic_events, gates

class GateReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="gate",
                                 class_type=self)
        super().__init__(*args, **kwargs)