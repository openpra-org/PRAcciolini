from typing import Set

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.event_tree.functional_event import FunctionalEventDefinition
from pracciolini.grammar.openpsa.xml.event_tree.initial_state import InitialStateDefinition
from pracciolini.grammar.openpsa.xml.event_tree.sequence import SequenceDefinition
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable, Role, Label, Attributes


class EventTreeDefinition(XMLSerializable):
    def __init__(self, name, functional_events=None, sequences=None, initial_state=None, role=None, label=None, attributes=None):
        self.name: str = name
        self.functional_events: Set[FunctionalEventDefinition] = functional_events if functional_events else set()
        self.sequences: Set[SequenceDefinition] = sequences if sequences else set()
        self.initial_state: InitialStateDefinition = initial_state if initial_state else None
        # undeveloped attrs
        self.role = Role(role) if role else None
        self.label = Label(label) if label else None
        self.attributes = Attributes(attributes) if attributes else None

    def to_xml(self):
        element = etree.Element("define-event-tree")
        element.set("name", self.name)
        if self.role:
            element.set("role", self.role.to_xml())
        if self.label:
            element.append(self.label.to_xml())
        if self.attributes:
            element.append(self.attributes.to_xml())

        for fe in self.functional_events:
            element.append(fe.to_xml())
        for seq in self.sequences:
            element.append(seq.to_xml())
        if self.initial_state:
            element.append(self.initial_state.to_xml())
        return element


    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'EventTreeDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("name is missing in define-event-tree")
        event_tree: EventTreeDefinition = EventTreeDefinition(name=name)
        # parse functional-event definition list
        functional_events_xml = root.findall("define-functional-event")
        for functional_event_xml in functional_events_xml:
            functional_event: FunctionalEventDefinition = FunctionalEventDefinition.from_xml(functional_event_xml)
            event_tree.functional_events.add(functional_event)
            print(functional_event)

        # parse sequence definition list
        sequences_xml = root.findall("define-sequence")
        for sequence_xml in sequences_xml:
            sequence: SequenceDefinition = SequenceDefinition.from_xml(sequence_xml)
            event_tree.sequences.add(sequence)
            print(sequence)

        # parse initial-state
        initial_state_xml = root.find("initial-state")
        if initial_state_xml is None:
            raise lxml.etree.ParserError("initial-state is missing in define-event-tree")
        initial_state: InitialStateDefinition = InitialStateDefinition.from_xml(initial_state_xml)
        event_tree.initial_state = initial_state

        return event_tree

    def __str__(self):
        return (f"\nevent-tree-definition: name: {self.name}"
                f"\n\tfunctional-events: {len(self.functional_events)}"
                f"\n\tsequences: {len(self.sequences)}"
                f"\n\tinitial-state: {self.initial_state}")