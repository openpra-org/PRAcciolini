from typing import List, Set, Tuple
from collections import OrderedDict

from lxml.etree import Element
import tensorflow as tf

from pracciolini.core.decorators import translation
from pracciolini.grammar.canopy.io.OpCode import OpCode
from pracciolini.grammar.canopy.model.ops.sampler import generate_bernoulli
from pracciolini.grammar.canopy.model.subgraph import SubGraph
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.validate import read_openpsa_xml
from pracciolini.grammar.openpsa.xml.expression.logical import NotExpression, LogicalExpression
from pracciolini.grammar.openpsa.xml.fault_tree import GateReference
from pracciolini.grammar.openpsa.xml.reference import BasicEventReference, HouseEventReference


def build_operation(gate_xml, subgraph, input_tensors_map):
    gate = OpsaMefXmlRegistry.instance().build(gate_xml)
    opcode = subgraph.supported_logical_opcodes.get(gate.type, None)
    if opcode is None:
        print(f"ignoring unsupported gate of type: {gate.type}")
        return None, input_tensors_map
    children_names = [child.name for child in gate.children[0].children]
    op_inputs: List[Tensor] = []
    for child in children_names:
        child_tensor = input_tensors_map.get(child, None)
        if child_tensor is None:
            raise TypeError(f"Attempted to use input tensor {child} but it does not exist in subgraph {subgraph}")
        op_inputs.append(child_tensor)
    constructed_op = subgraph.bitwise(opcode=opcode, operands=op_inputs, name=f"{gate.type}_{gate.name}")
    input_tensors_map[gate.name] = constructed_op
    return gate.name, input_tensors_map

def get_referenced_events(tree: Element) -> Set[str]:
    events: Set[str]= set()
    referenced_events_xml = tree.xpath("//event | //house-event | //basic-event | //gate")
    for event_ref_xml in referenced_events_xml:
        event_ref = OpsaMefXmlRegistry.instance().build(event_ref_xml)
        events.add(event_ref.name)
    return events

def get_event_definitions(tree: Element) -> Set[str]:
    events: Set[str]= set()
    referenced_events_xml = tree.xpath("//define-event | //define-house-event | //define-basic-event | //define-gate")
    for event_ref_xml in referenced_events_xml:
        event_ref = OpsaMefXmlRegistry.instance().build(event_ref_xml)
        events.add(event_ref.name)
    return events

def _build_event_tensor(event) -> Tensor:
    probs: tf.Tensor = tf.constant([float(event.value), float(event.value)], name=event.name) # needs to be at-least 2items for now
    tf_tensor = generate_bernoulli(probs=probs, count=128, bitpack_dtype=tf.uint8, dtype=tf.float64)
    print(tf_tensor)
    samples = Tensor(tf_tensor, name=event.name)
    return samples

def build_event_definition(xml, subgraph: SubGraph, inputs):
    event = OpsaMefXmlRegistry.instance().build(xml)
    if event.name in inputs:
        return event.name, inputs[event.name]
    samples = _build_event_tensor(event)
    subgraph.register_input(samples)
    inputs[event.name] = samples
    return event.name, samples

def build_gate_definition(xml, subgraph: SubGraph, inputs, operators):
    gate = OpsaMefXmlRegistry.instance().build(xml)
    if gate.name in operators:
        return gate.name, operators[gate.name]

    opcode = subgraph.supported_logical_opcodes.get(gate.type, None)
    if opcode is None:
        opcode = OpCode.BITWISE_OR
        print(f"replacing with OR, unsupported gate {gate.name} of type: {gate.type}")

    children_names = set()
    for child in gate.children[0].children:
        if isinstance(child, NotExpression):
            children_names.add(child.children[0].name)
        else:
            children_names.add(child.name)
    #children_names = set([child.name for child in gate.children[0].children])
    if not set(inputs.keys()).union(operators.keys()).issuperset(children_names):
        return None, None

    op_inputs: List[Tensor] = []
    for child in children_names:
        child_tensor = inputs.get(child, operators.get(child, None))
        if child_tensor is None:
            raise TypeError(f"Attempted to use input tensor {child} but it does not exist in subgraph {subgraph}")
        op_inputs.append(child_tensor)

    constructed_op = subgraph.bitwise(opcode=opcode, operands=op_inputs, name=f"{gate.type}_{gate.name}")
    operators[gate.name] = constructed_op

    return gate.name, constructed_op

def build_not_gate_definition(gate_def_xml, not_gate_ref, model_xml, subgraph: SubGraph, inputs, operators):
    return None, None
    #return gate.name, constructed_op

def build_gate_definition_recursive(gate_def_xml, model_xml, subgraph: SubGraph, inputs, operators):
    gate = OpsaMefXmlRegistry.instance().build(gate_def_xml)
    if gate.name in operators:
        return gate.name, operators[gate.name]

    opcode = subgraph.supported_logical_opcodes.get(gate.type, None)
    if opcode is None:
        opcode = OpCode.BITWISE_OR
        print(f"replacing with OR, unsupported gate {gate.name} of type: {gate.type}")

    child_references = set()
    for child in gate.children[0].children:
        if isinstance(child, GateReference):
            op_name, op = build_gate_definition_recursive(child, model_xml, subgraph, inputs, operators)
            child_references.add(op_name)
        elif isinstance(child, BasicEventReference) or isinstance(child, HouseEventReference):
            op_name, op = build_event_definition(model_xml, subgraph, inputs)
            child_references.add(op_name)
        elif isinstance(child, NotExpression):
            op_name, op = build_not_gate_definition(gate_def_xml, child, model_xml, subgraph, inputs, operators)
            child_references.add(op_name)
        else:
            raise ValueError(f"encountered unknown reference type {child} while building {gate.type}-gate {gate.name}")

    op_inputs: List[Tensor] = []
    for child in child_references:
        child_tensor = inputs.get(child, operators.get(child, None))
        if child_tensor is None:
            raise TypeError(f"Attempted to use input tensor {child} but it does not exist in subgraph {subgraph}")
        op_inputs.append(child_tensor)

    constructed_op = subgraph.bitwise(opcode=opcode, operands=op_inputs, name=f"{gate.type}_{gate.name}")
    operators[gate.name] = constructed_op

    return gate.name, constructed_op

def get_event_references_and_definitions(tree: Element):
    referenced_events = get_referenced_events(tree)
    event_definitions = get_event_definitions(tree)
    return referenced_events, event_definitions

def build_definitions(tree: Element, subgraph: SubGraph):
    referenced_events, event_definitions = get_event_references_and_definitions(tree)
    referenced_but_not_defined = referenced_events - event_definitions
    print(f"\nundefined references: {referenced_but_not_defined}")
    defined_but_not_referenced = event_definitions - referenced_events
    print(f"unreferenced definitions: {defined_but_not_referenced}")
    unvisited_event_definitions = event_definitions.copy()
    print(f"unvisited definitions: {unvisited_event_definitions}")

    inputs = OrderedDict()
    operators = OrderedDict()
    outputs = OrderedDict()

    be_defs_xml = tree.xpath("//define-basic-event | //define-house-event")
    for be_xml in be_defs_xml:
        name, tensor = build_event_definition(be_xml, subgraph, inputs)
        if name is not None and name in unvisited_event_definitions:
            unvisited_event_definitions.remove(name)

    while len(unvisited_event_definitions) > 0:
        for gate_def_xml in tree.xpath("//define-gate"):
            op_name, op = build_gate_definition(gate_def_xml, subgraph, inputs, operators)
            if op_name is not None and op_name in unvisited_event_definitions:
                unvisited_event_definitions.remove(op_name)
            #print(f"remaining unvisited: {unvisited_event_definitions}")

    return inputs, operators, outputs

def opsamef_ft_xml_to_canopy_subgraph(ft_xml: Element, model_xml: Element) -> SubGraph:
    subgraph: SubGraph = SubGraph()
    inputs = OrderedDict()
    operators = OrderedDict()

    gate_def_xmls = ft_xml.xpath("//define-gate")
    for gate_def_xml in gate_def_xmls:
        op_name, op = build_gate_definition_recursive(gate_def_xml, model_xml, subgraph, inputs, operators)
        subgraph.register_output(op)

    subgraph.prune()

    return subgraph

@translation('opsamef_ft_xml_file', 'canopy_subgraphs')
def opsamef_fts_xml_to_canopy_subgraphs(file_path: str) -> Set[SubGraph]:
    subgraphs: Set[SubGraph] = set()
    xml_data = read_openpsa_xml(file_path)
    for ft_def_xml in xml_data.xpath("//define-fault-tree"):
        subgraph = opsamef_ft_xml_to_canopy_subgraph(ft_def_xml, xml_data)
        subgraphs.add(subgraph)
    return subgraphs

## note: we are bit-packing the samples, which is fine, but it only gives us linear speed up
## rather, we should be using the bit-fields as gate inputs. for example, if we see that an event tree has 12 functional
# events, we should use a 16-bit dtype to encode the state of each of the functional events (what we have been doing
# with PLA).

