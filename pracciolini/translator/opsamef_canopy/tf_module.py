import timeit
from dataclasses import dataclass
from typing import Any, Dict, List

import tensorflow as tf
from lxml.etree import Element

from pracciolini.core.decorators import translation
from pracciolini.grammar.canopy.module.bitwise import bitwise_or, bitwise_and, bitwise_not, bitwise_xor, bitwise_xnor, \
    bitwise_nor, bitwise_nand, bitwise_atleast
from pracciolini.grammar.canopy.module.broadcast_sampler import LogicTreeBroadcastSampler
from pracciolini.grammar.canopy.sampler.sizer import BatchSampleSizeOptimizer
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.validate import read_openpsa_xml
from pracciolini.grammar.openpsa.xml.define_event import BasicEventDefinition
from pracciolini.grammar.openpsa.xml.fault_tree import GateReference, GateDefinition


class OpsamefCanopyFaultTreeTranslator:

    def __init__(self, model_xml_filepath: Element):
        super().__init__()
        self._model_xml_filepath = model_xml_filepath
        self._model_xml = read_openpsa_xml(self._model_xml_filepath)

        self._input_event_definitions = dict()
        self._gates = dict()
        self._fault_tree_definitions = dict()

        self._fault_tree_tops = {}
        #self._build_input_definitions()
        #self._output_shape = None

        self._basic_event_indices: Dict[str, int] = dict()
        self._basic_event_probs: List[float] = []

        self._gate_indices: Dict[str, int] = dict()
        self._basic_event_probs: List[float] = []

    def _build_input_definitions(self):
        defined_events_xml = self._model_xml.xpath("//define-house-event | //define-basic-event")
        for event_ref_xml in defined_events_xml:
            event_ref = OpsaMefXmlRegistry.instance().build(event_ref_xml)
            #print(event_ref.name)
            #events.add(event_ref.name)

    @staticmethod
    def __lookup_op_by_gate_type(gate_type: str):
        match gate_type:
            case "or": return bitwise_or
            case "and": return bitwise_and
            case "not": return bitwise_not
            case "xor": return bitwise_xor
            case "xnor": return bitwise_xnor
            case "nor": return bitwise_nor
            case "nand": return bitwise_nand
            case "atleast": return bitwise_atleast
            case _: raise Exception(f"Unable to handle gate of type {gate_type}.")


    def __lookup_item_by_name(self, name_: str, dict_, tag) -> tuple[bool, int | BasicEventDefinition | GateDefinition]:
        # return the event if already built
        if name_ in dict_:
            return True, dict_[name_]
        # if not, find it in the xml, throw an exception if it is not found
        xml_ = self._model_xml.xpath(f"//{tag}[@name='{name_}']")
        if xml_ is None or len(xml_) == 0:
            raise Exception(f"No definition found for {xml_}")
        if len(xml_) > 1:
            print(f"multiple definitions found for {xml_}, using the first one")
        return False, OpsaMefXmlRegistry.instance().build(xml_[0])

    def _get_basic_event(self, basic_event_def_name: str) -> int:
        event_found, event_def = self.__lookup_item_by_name(basic_event_def_name, self._basic_event_indices, "define-basic-event")
        if event_found:
            return event_def

        ## event was not found, we will build it from the XML metadata
        num_basic_events = len(self._basic_event_probs)
        assert len(self._basic_event_probs) == len(self._basic_event_indices.keys())
        # add a new key to the basic event dictionary that maps the event name to an index on a vector with their probabilities
        self._basic_event_indices[basic_event_def_name] = num_basic_events
        self._basic_event_probs.append(float(event_def.value))
        assert len(self._basic_event_probs) == len(self._basic_event_indices.keys())
        assert len(self._basic_event_probs) == num_basic_events + 1
        assert self._basic_event_probs[num_basic_events] == float(event_def.value)
        assert self._basic_event_indices[basic_event_def_name] == num_basic_events
        #print(f"basic_event_probs: {self._basic_event_probs}")
        #print(f"basic_event_indices: {self._basic_event_indices.items()}")
        return num_basic_events


    def _get_gate(self, gate_def_name: str):
        event_found, gate_def = self.__lookup_item_by_name(gate_def_name, self._gates, "define-gate")
        if event_found:
            return gate_def

        # collect the gate metadata
        gate_type = gate_def.type
        gate_num_inputs = len(gate_def.referenced_events)

        # build the gate input recursively
        print(f"gate: {gate_def.name}, type: {gate_type}, children: {len(gate_def.referenced_events)}")
        basic_event_refs, gate_refs = gate_def.referenced_events_by_type

        be_ref_tensor_input_indices = [self._get_basic_event(be_ref.name) for be_ref in basic_event_refs]
        gate_inputs = [self._get_gate(gate_ref.name) for gate_ref in gate_refs]

        # apply the operator for the gate, build the function, and store it
        gate_op = self.__lookup_op_by_gate_type(gate_type)

        def _logic_fn(inputs):
            be_inputs_ = tf.gather(inputs, be_ref_tensor_input_indices)
            be_sub_gate_ = gate_op(be_inputs_)
            other_gate_outputs_ = tf.parallel_stack(gate_inputs)
            output_ = tf.bitwise.bitwise_xor(be_sub_gate_, other_gate_outputs_)
            return output_

        self._gates[gate_def_name] = _logic_fn

        return gate_op


    def translate(self):
        logic_fn_ = self.build_logic_fn()
        num_inputs_ = len(self._basic_event_probs)
        input_probs_ = self._basic_event_probs
        num_outputs_ = 1
        return input_probs_, num_inputs_, logic_fn_, num_outputs_


    def _build_fault_tree_logic_fxns(self):
        ft_logic_fxns_ = []
        for ft_def_xml in self._model_xml.xpath("//define-fault-tree"):
            ft = OpsaMefXmlRegistry.instance().build(ft_def_xml)
            ft_top = self._get_gate(ft.gates[0].name)
            ft_logic_fxns_.append(ft_top)
            break
        return ft_logic_fxns_


    def build_logic_fn(self):
        return self._build_fault_tree_logic_fxns()[0]



@translation('opsamef_xml_file', 'tf_module')
def opsamef_xml_to_tensorflow_module(file_path: str) -> Any:
    translator = OpsamefCanopyFaultTreeTranslator(file_path)
    return translator.translate()


if __name__ == '__main__':
    #xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/demo/gas_leak.xml"


    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-openpsa-models/models/Aralia/das9701.xml"
    input_probs, num_inputs, logic_fn, num_outputs = opsamef_xml_to_tensorflow_module(xml_file_path)


    print("input_probs:", input_probs)
    print("num_inputs:", num_inputs)
    print("logic_fn:", logic_fn)
    print("num_outputs:", num_outputs)

    sampler_dtype = tf.float32
    bitpack_dtype = tf.uint8
    optimizer = BatchSampleSizeOptimizer(
        num_events=num_inputs,
        max_bytes=int(1.5 * 2 ** 32),  # 1.8 times 4 GiB
        sampled_bits_per_event_range=(1e6, None),
        sampler_dtype=sampler_dtype,
        bitpack_dtype=bitpack_dtype,
        batch_size_range=(1, None),
        sample_size_range=(1, None),
        total_batches_range=(1, None),
        max_iterations=10000,
        tolerance=1e-8,
    )

    optimizer.optimize()
    sample_sizer = optimizer.get_results()

    sampler = LogicTreeBroadcastSampler(
        logic_fn=logic_fn,
        num_inputs=sample_sizer['num_events'],
        num_outputs=num_outputs,
        num_batches=sample_sizer['total_batches'],
        batch_size=sample_sizer['batch_size'],
        sample_size=sample_sizer['sample_size'],
        bitpack_dtype=bitpack_dtype,
        sampler_dtype=sampler_dtype,
        acc_dtype=sampler_dtype
    )

    count = 2
    t = timeit.Timer(lambda: sampler.tally(input_probs))
    times = t.timeit(number=count)
    print(f"total_time (s): {times}")
    print(f"avg_time (s): {times / float(count)}")
    print(sampler.tally(input_probs))