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

    @staticmethod
    def _build_op_0be_1gate(input_gate_inputs_):
        print(f"building op-0be-1gate for {input_gate_inputs_}")
        def _graph_op_0be_1gate_(inputs):
            gate_op = input_gate_inputs_[0]
            return gate_op(inputs)
        return _graph_op_0be_1gate_

    @staticmethod
    def _build_op_1be_0gate(be_tensor_indices_):
        print(f"building op-1be-0gate for {be_tensor_indices_}")
        def _graph_op_1be_0gate_(inputs):
            be_inputs_ = tf.gather(inputs, be_tensor_indices_) # tensor with shape (1, batch_size, sample_size)
            return be_inputs_[0, :, :] # tensor with shape (batch_size, sample_size)
        return _graph_op_1be_0gate_

    @staticmethod
    def _build_op_geq_2be_0gate(bitwise_nary_op_, be_tensor_indices_):
        print(f"building op-geq-2be-0gate for {be_tensor_indices_}, op: {bitwise_nary_op_}")
        def _graph_op_geq_2be_0gate_(inputs):
            be_inputs_ = tf.gather(inputs, be_tensor_indices_) # tensor with shape (n, batch_size, sample_size)
            return bitwise_nary_op_(be_inputs_) # tensor with shape (batch_size, sample_size)
        return _graph_op_geq_2be_0gate_

    @staticmethod
    def _build_op_0be_geq_2gate(bitwise_nary_op_, input_gate_inputs_):
        print(f"building op-0be-geq-2gate for {input_gate_inputs_}, op: {bitwise_nary_op_}")
        def _graph_op_0be_geq_2gate(inputs):
            # Apply each function in op_inputs to inputs
            input_gate_outputs_ = [op(inputs) for op in input_gate_inputs_]  # Each output has shape [batch_size, sample_size]

            # Stack outputs into a tensor of shape [num_ops, batch_size, sample_size]
            input_gate_outputs_stacked_ = tf.stack(input_gate_outputs_, axis=0)

            # # Use tf.foldl to reduce over the outputs using new_op
            # result = tf.foldl(
            #     lambda accum, elem: bitwise_nary_op_(accum, elem),
            #     input_gate_outputs_stacked_[1:],  # Start from the second element if outputs_stacked has more than one element
            #     initializer=input_gate_outputs_stacked_[0]
            # )
            return bitwise_nary_op_(input_gate_outputs_stacked_)
            #return bitwise_nary_op_(be_inputs_) # tensor with shape (batch_size, sample_size)
        return _graph_op_0be_geq_2gate

    @staticmethod
    def _build_op_geq_1be_geq_1gate(bitwise_nary_op_, be_tensor_indices_, input_gate_inputs_):
        print(f"building op-geq-1be-geq-1gate for BEs: {be_tensor_indices_}, Gates: {input_gate_inputs_}, op: {bitwise_nary_op_}")
        def _graph_op_geq_1be_geq_1gate(inputs):
            be_inputs_ = tf.gather(inputs, be_tensor_indices_)
            #print("be_inputs_.shape", be_inputs_.shape)
            # Apply each function in op_inputs to inputs
            input_gate_outputs_ = [op(inputs) for op in input_gate_inputs_]  # Each output has shape [batch_size, sample_size]
            # Stack outputs into a tensor of shape [num_ops, batch_size, sample_size]
            input_gate_outputs_stacked_ = tf.stack(input_gate_outputs_, axis=0)
            #print("input_gate_outputs_stacked_.shape", input_gate_outputs_stacked_.shape)
            gate_inputs_ = tf.concat([be_inputs_, input_gate_outputs_stacked_], axis=0)
            #print("gate_inputs_.shape", gate_inputs_.shape)
            # Use tf.foldl to reduce over the outputs using new_op
            # result = tf.foldl(
            #     lambda accum, elem: bitwise_nary_op_(accum, elem),
            #     gate_inputs_[1:],  # Start from the second element if outputs_stacked has more than one element
            #     initializer=gate_inputs_[0]
            # )
            return bitwise_nary_op_(gate_inputs_)
        return _graph_op_geq_1be_geq_1gate

    def _get_gate(self, gate_def_name: str):
        event_found, gate_def = self.__lookup_item_by_name(gate_def_name, self._gates, "define-gate")
        if event_found:
            return gate_def

        # collect the gate metadata
        gate_type = gate_def.type
        #gate_num_inputs = len(gate_def.referenced_events)

        # build the gate input recursively
        #print(f"gate: {gate_def.name}, type: {gate_type}, children: {len(gate_def.referenced_events)}")
        basic_event_refs, gate_refs = gate_def.referenced_events_by_type

        be_ref_tensor_input_indices = list(set([self._get_basic_event(be_ref.name) for be_ref in basic_event_refs]))
        gate_inputs = list(set([self._get_gate(gate_ref.name) for gate_ref in gate_refs]))

        num_be_inputs = len(be_ref_tensor_input_indices)
        num_gate_inputs = len(gate_inputs)
        # apply the operator for the gate, build the function, and store it
        gate_op = self.__lookup_op_by_gate_type(gate_type)

        logic_fn_ = None
        if num_gate_inputs >= 1 and num_be_inputs >= 1:
            logic_fn_ = self._build_op_geq_1be_geq_1gate(bitwise_nary_op_=gate_op,
                                                        be_tensor_indices_=be_ref_tensor_input_indices,
                                                        input_gate_inputs_=gate_inputs)
        elif num_gate_inputs >= 2 and num_be_inputs == 0:
            logic_fn_ = self._build_op_0be_geq_2gate(bitwise_nary_op_=gate_op,
                                                    input_gate_inputs_=gate_inputs)
        elif num_gate_inputs == 0 and num_be_inputs >= 2:
            logic_fn_ = self._build_op_geq_2be_0gate(bitwise_nary_op_=gate_op,
                                                    be_tensor_indices_=be_ref_tensor_input_indices)
        elif num_gate_inputs == 1 and num_be_inputs == 0:
            logic_fn_ = self._build_op_0be_1gate(input_gate_inputs_=gate_inputs)
        elif num_gate_inputs == 0 and num_be_inputs == 1:
            logic_fn_ = self._build_op_1be_0gate(be_tensor_indices_=be_ref_tensor_input_indices)
        elif num_gate_inputs == 0 and num_be_inputs == 0:
            raise Exception(f"Unable to handle gate of type {gate_type} with no inputs.")
        else:
            raise Exception(f"Unable to handle handle case with num_be_inputs={num_be_inputs} and num_gate_inputs={num_gate_inputs}.")
        #print(f"be_ref_tensor_input_indices: {be_ref_tensor_input_indices}")
        #print(f"gathered be_inputs: {be_inputs_.shape}, {be_inputs_}")
        #print(f"inputs: {inputs.shape}, {inputs}")

        self._gates[gate_def_name] = logic_fn_

        return logic_fn_


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
    tf.config.run_functions_eagerly(False)

    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-openpsa-models/models/Aralia/das9701.xml"
    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-pwr-openpsa-model/models/converted_EQK-BIN1_et_Grp-1_24-02-26_15-57-10.xml"

    #input_probs, num_inputs, logic_fn, num_outputs = opsamef_xml_to_tensorflow_module(xml_file_path)
    input_probs, num_inputs, logic_, num_outputs = OpsamefCanopyFaultTreeTranslator(xml_file_path).translate()

    print("input_probs:", input_probs)
    print("num_inputs:", num_inputs)
    print("logic_fn:", logic_)
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
        logic_fn=logic_,
        num_inputs=sample_sizer['num_events'],
        num_outputs=num_outputs,
        num_batches=sample_sizer['total_batches'],
        batch_size=sample_sizer['batch_size'],
        sample_size=sample_sizer['sample_size'],
        bitpack_dtype=bitpack_dtype,
        sampler_dtype=sampler_dtype,
        acc_dtype=sampler_dtype
    )

    # @tf.function(jit_compile=True)
    # def run():
    #     print(sampler.generate(input_probs).shape)
    #     return logic_fn(sampler.generate(input_probs))

    print(sampler.tally(input_probs))

    #run()
    # count = 2
    # t = timeit.Timer(lambda: sampler.tally(input_probs))
    # times = t.timeit(number=count)
    # print(f"total_time (s): {times}")
    # print(f"avg_time (s): {times / float(count)}")
    #print(sampler.tally_from_samples(sampler.eval_fn(logic_, input_probs)))