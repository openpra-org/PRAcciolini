from typing import Set, Optional

import lxml.etree
from lxml import etree

from pracciolini.grammar.openpsa.xml.event_tree.event_tree import EventTreeDefinition
from pracciolini.grammar.openpsa.xml.event_tree.initiating_event import InitiatingEventDefinition
from pracciolini.grammar.openpsa.xml.fault_tree.fault_tree import FaultTreeDefinition
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable
from pracciolini.grammar.openpsa.xml.model_data.model_data import ModelData


class OpsaMef(XMLSerializable):
    def __init__(self,
                 initiating_events: Optional[Set[InitiatingEventDefinition]] = None,
                 event_trees: Optional[Set[EventTreeDefinition]] = None,
                 fault_trees: Optional[Set[FaultTreeDefinition]] = None,
                 model_data: Optional[ModelData] = None,
        ):
        self.model_data: Optional[ModelData] = model_data
        self.initiating_events: Set[InitiatingEventDefinition] = initiating_events if initiating_events is not None else set()
        self.event_trees: Set[EventTreeDefinition] = event_trees if event_trees is not None else set()
        self.fault_trees: Set[FaultTreeDefinition] = fault_trees if fault_trees is not None else set()

    def to_xml(self):
        opsa_mef_elem = etree.Element("opsa-mef")

        for initiating_event in self.initiating_events:
            opsa_mef_elem.append(initiating_event.to_xml())

        for event_tree in self.event_trees:
            opsa_mef_elem.append(event_tree.to_xml())

        for fault_tree in self.fault_trees:
            opsa_mef_elem.append(fault_tree.to_xml())

        # finally, add the model data
        opsa_mef_elem.append(self.model_data.to_xml())

        return opsa_mef_elem

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'OpsaMef':
        # parse initiating-event definition list
        ie_set: Set[InitiatingEventDefinition] = set()
        initiating_events_xml = root.findall("define-initiating-event")
        for initiating_event_xml in initiating_events_xml:
            initiating_event: InitiatingEventDefinition = InitiatingEventDefinition.from_xml(initiating_event_xml)
            ie_set.add(initiating_event)

        # parse event-tree definition list
        et_set: Set[EventTreeDefinition] = set()
        event_tree_xmls = root.findall("define-event-tree")
        for event_tree_xml in event_tree_xmls:
            event_tree: EventTreeDefinition = EventTreeDefinition.from_xml(event_tree_xml)
            et_set.add(event_tree)

        # parse fault-tree definition list
        ft_set: Set[FaultTreeDefinition] = set()
        fault_tree_xmls = root.findall("define-fault-tree")
        for fault_tree_xml in fault_tree_xmls:
            fault_tree: FaultTreeDefinition = FaultTreeDefinition.from_xml(fault_tree_xml)
            ft_set.add(fault_tree)

        # parse the model-data
        model_data: ModelData = ModelData.from_xml(root.find("model-data"))

        return cls(initiating_events=ie_set, event_trees=et_set, fault_trees=ft_set, model_data=model_data)

    def __str__(self):
        return (f"\ninitiating-events: {len(self.initiating_events)}"
                f"\nevent-trees: {len(self.event_trees)}"
                f"\nfault-trees: {len(self.fault_trees)}"
                f"\nmodel-data: {self.model_data}")