from typing import Dict

from lxml.etree import Element

from pracciolini.core.decorators import translation
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.validate import read_openpsa_xml

def build_events_map(tree: Element) -> Dict[str, str]:
    events: Dict[str, str] = dict()
    event_defs_xml = tree.xpath("//define-initiating-event | //define-basic-event | //define-house-event")
    for event_xml in event_defs_xml:
        event = OpsaMefXmlRegistry.instance().build(event_xml)
        events[event.name] = event.value
    return events

@translation('opsamef_xml', 'expr')
def opsamef_xml_to_expr(file_path: str) -> str:
    try:
        xml_data = read_openpsa_xml(file_path)
        events_map = build_events_map(xml_data)
        print(f"total: {len(events_map.keys())}")
        return str(len(events_map.keys()))
    except Exception as e:
        print(f"An error occurred during translation: {e}")
    return "WIP"