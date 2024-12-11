from typing import List
from collections import OrderedDict

from lxml.etree import Element
import tensorflow as tf

from pracciolini.grammar.canopy.io.OpCode import OpCode
from pracciolini.grammar.canopy.model.subgraph import SubGraph
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.xml.expression.logical import NotExpression
from pracciolini.translator.opsamef_canopy.input_tensor_builder import bit_pack_samples

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

def get_referenced_events(tree: Element):
    events = set()
    referenced_events_xml = tree.xpath("//event | //house-event | //basic-event | //gate")
    for event_ref_xml in referenced_events_xml:
        event_ref = OpsaMefXmlRegistry.instance().build(event_ref_xml)
        events.add(event_ref.name)
    return events

def get_event_definitions(tree: Element):
    events = set()
    referenced_events_xml = tree.xpath("//define-event | //define-house-event | //define-basic-event | //define-gate")
    for event_ref_xml in referenced_events_xml:
        event_ref = OpsaMefXmlRegistry.instance().build(event_ref_xml)
        events.add(event_ref.name)
    return events

def build_basic_event_definition(xml, subgraph: SubGraph, inputs):
    be = OpsaMefXmlRegistry.instance().build(xml)
    if be.name in inputs:
        return be.name, inputs[be.name]
    be_samples = Tensor(tf.constant(bit_pack_samples(float(be.value)), name=be.name), name=be.name)
    subgraph.register_input(be_samples)
    inputs[be.name] = be_samples
    return be.name, be_samples

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
        name, tensor = build_basic_event_definition(be_xml, subgraph, inputs)
        if name is not None and name in unvisited_event_definitions:
            unvisited_event_definitions.remove(name)

    while len(unvisited_event_definitions) > 0:
        for gate_def_xml in tree.xpath("//define-gate"):
            op_name, op = build_gate_definition(gate_def_xml, subgraph, inputs, operators)
            if op_name is not None and op_name in unvisited_event_definitions:
                unvisited_event_definitions.remove(op_name)
            #print(f"remaining unvisited: {unvisited_event_definitions}")

    return inputs, operators, outputs

def opsamef_ft_xml_to_canopy_subgraph(ft_xml: Element) -> SubGraph:
    subgraph: SubGraph = SubGraph()
    return subgraph

## note: we are bit-packing the samples, which is fine, but it only gives us linear speed up
## rather, we should be using the bit-fields as gate inputs. for example, if we see that an event tree has 12 functional
# events, we should use a 16-bit dtype to encode the state of each of the functional events (what we have been doing
# with PLA).

