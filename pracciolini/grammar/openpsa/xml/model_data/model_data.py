from typing import Set

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable
from pracciolini.grammar.openpsa.xml.model_data.event import BasicEventDefinition, HouseEventDefinition
from pracciolini.grammar.openpsa.xml.model_data.event import ParameterDefinition


class ModelData(XMLSerializable):
    def __init__(self, house_events=None, basic_events=None, parameters=None):
        self.house_events = [HouseEventDefinition(**he) for he in house_events] if house_events else set()
        self.basic_events: Set[BasicEventDefinition] = set(BasicEventDefinition(**be) for be in basic_events) if basic_events else set()
        self.parameters = [ParameterDefinition(**p) for p in parameters] if parameters else set()

    def to_xml(self):
        model_data_elem = etree.Element("model-data")
        for house_event in self.house_events:
            model_data_elem.append(house_event.to_xml())
        for basic_event in self.basic_events:
            model_data_elem.append(basic_event.to_xml())
        for parameter in self.parameters:
            model_data_elem.append(parameter.to_xml())
        return model_data_elem

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'ModelData':
        model_data: ModelData = ModelData()
        if root is None:
            return model_data
        basic_events_xml = root.findall("define-basic-event")
        for basic_event_xml in basic_events_xml:
            basic_event: BasicEventDefinition = BasicEventDefinition.from_xml(basic_event_xml)
            model_data.basic_events.add(basic_event)
        return model_data

    def __str__(self):
        return (f"\n\tbasic-events: {len(self.basic_events)}"
                f"\n\thouse-events: {len(self.house_events)}"
                f"\n\tparameters: {len(self.parameters)}")