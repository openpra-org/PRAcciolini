from typing import List

from pracciolini.grammar.openpsa.xml.fault_tree import FaultTreeDefinition
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class OpsaMef(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="opsa-mef",
                                 class_type=self,
                                 attrs={"name"},
                                 children={"define-initiating-event", "define-event-tree", "define-fault-tree", "define-rule", "model-data"})
        super().__init__(*args, **kwargs)

    @property
    def fault_trees(self):
        fault_trees_: List[FaultTreeDefinition] = []
        for child in self.children:
            if isinstance(child, FaultTreeDefinition):
                fault_trees_.append(child)
        return fault_trees_