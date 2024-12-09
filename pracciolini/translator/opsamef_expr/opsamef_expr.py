from typing import Dict, Tuple

from lxml.etree import Element
from pyeda.boolalg.expr import exprvar

from pracciolini.core.decorators import translation
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.validate import read_openpsa_xml
from pracciolini.grammar.openpsa.xml.fault_tree import FaultTreeDefinition


def build_fault_tree(xml_tree: Element) -> Tuple[str, str]:
    fault_tree: FaultTreeDefinition = OpsaMefXmlRegistry.instance().build(xml_tree)
    return fault_tree.name, fault_tree.to_expr() #str(fault_tree)

def build_fault_trees_map(xml_tree: Element) -> Dict[str, str]:
    ft_defs_xml = xml_tree.xpath("//define-fault-tree")
    fault_trees: Dict[str, str] = dict()
    for ft_xml in ft_defs_xml:
        ft_name, ft_expr = build_fault_tree(ft_xml)
        fault_trees[ft_name] = ft_expr
    return fault_trees

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

        expr_vars = exprvar(tuple(events_map.keys()))
        print(f"events: {expr_vars.names}")

        fault_trees_map = build_fault_trees_map(xml_data)
        print(f"fault trees: {(fault_trees_map.keys())}")

        return str(len(events_map.keys()) + len(fault_trees_map.keys()))
    except Exception as e:
        print(f"An error occurred during translation: {e}")
    return "WIP"