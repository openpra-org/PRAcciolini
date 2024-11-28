from typing import Set, Optional

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable
from pracciolini.grammar.openpsa.xml.model_data.event import BasicEventDefinition, HouseEventDefinition
from pracciolini.grammar.openpsa.xml.model_data.event import ParameterDefinition


class ModelData(XMLSerializable):
    def __init__(self,
                 basic_events: Optional[Set[BasicEventDefinition]]=None,
                 house_events: Optional[Set[HouseEventDefinition]]=None,
                 parameters: Optional[Set[ParameterDefinition]]=None
        ):
        self.house_events: Set[HouseEventDefinition] = house_events if house_events is not None else set()
        self.basic_events: Set[BasicEventDefinition] = basic_events if basic_events is not None else set()
        self.parameters: Set[ParameterDefinition] = parameters if parameters is not None else set()

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
        if root is None:
            raise lxml.etree.ParserError("model-data tag does not exist")
        # parse the basic-events
        basic_events: Set[BasicEventDefinition] = set()
        basic_events_xml = root.findall("define-basic-event")
        for basic_event_xml in basic_events_xml:
            basic_event: BasicEventDefinition = BasicEventDefinition.from_xml(basic_event_xml)
            basic_events.add(basic_event)

        # parse the house-events
        house_events: Set[HouseEventDefinition] = set()
        house_events_xml = root.findall("define-house-event")
        for house_event_xml in house_events_xml:
            house_event: HouseEventDefinition = HouseEventDefinition.from_xml(house_event_xml)
            house_events.add(house_event)

        return cls(basic_events=basic_events, house_events=house_events)

    def __str__(self):
        return (f"\n\tbasic-events: {len(self.basic_events)}"
                f"\n\thouse-events: {len(self.house_events)}"
                f"\n\tparameters: {len(self.parameters)}")