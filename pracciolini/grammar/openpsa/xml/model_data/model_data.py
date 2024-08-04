from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable
from pracciolini.grammar.openpsa.xml.model_data.event import BasicEventDefinition, HouseEventDefinition
from pracciolini.grammar.openpsa.xml.model_data.event import ParameterDefinition


class ModelData(XMLSerializable):
    def __init__(self, house_events=None, basic_events=None, parameters=None):
        self.house_events = [HouseEventDefinition(**he) for he in house_events] if house_events else []
        self.basic_events = [BasicEventDefinition(**be) for be in basic_events] if basic_events else []
        self.parameters = [ParameterDefinition(**p) for p in parameters] if parameters else []

    def to_xml(self):
        model_data_elem = etree.Element("model-data")
        for house_event in self.house_events:
            model_data_elem.append(house_event.to_xml())
        for basic_event in self.basic_events:
            model_data_elem.append(basic_event.to_xml())
        for parameter in self.parameters:
            model_data_elem.append(parameter.to_xml())
        return model_data_elem
