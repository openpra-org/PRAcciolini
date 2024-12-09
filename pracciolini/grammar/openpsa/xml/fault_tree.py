from sys import exception
from typing import Dict

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

    def to_expr(self) -> Dict[str, str]:
        expr_map: Dict[str, str] = dict()
        try:
            for child in self.children:
                expr_map[child.name] = child.to_expr()
        except:
            pass
        for key, value in expr_map.items():
            print(key, value)
        return expr_map


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

    def to_expr(self) -> str:
        try:
            return (self.children[0]).to_expr()
        except Exception as e:
            return f"{self.name} to_expr error: {str(e)}"
    # @property
    # def type(self) -> str | None:
    #     try:
    #         return self.children[0].tag
    #     except:
    #         return None
    #
    # @property
    # def symbol(self) -> str | None:
    #     try:
    #         match self.type:
    #             case "and": return "&"
    #             case "or": return "|"
    #             case "not": return "~"
    #             case "xor": return "^"
    #             case _: return "?"
    #     except:
    #         return "?"



class GateReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="gate",
                                 class_type=self)
        super().__init__(*args, **kwargs)