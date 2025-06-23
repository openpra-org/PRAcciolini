import contextlib
import os
import timeit
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import tensorflow as tf
from lxml.etree import Element

from pracciolini.core.decorators import translation
from pracciolini.grammar.canopy.module.bitwise import bitwise_or, bitwise_and, bitwise_not, bitwise_xor, bitwise_xnor, \
    bitwise_nor, bitwise_nand, bitwise_atleast, efficient_bitwise_or, efficient_bitwise_and
from pracciolini.grammar.canopy.module.broadcast_sampler import LogicTreeBroadcastSampler
from pracciolini.grammar.canopy.module.sampler import Sampler
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

        self._basic_event_indices = dict()
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
            case "or": return efficient_bitwise_or
            case "and": return efficient_bitwise_and
            case "not": return bitwise_not
            case "xor": return bitwise_xor
            case "xnor": return bitwise_xnor
            case "nor": return bitwise_nor
            case "nand": return bitwise_nand
            case "atleast": return efficient_bitwise_or
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


    def _build_basic_event_subset_sampler(self, probs_: List[float], be_name_: Optional[str] = None):

        def _graph_op_sample_bernoulli_(batch_size, sample_size, sampler_dtype, bitpack_dtype):
            probs = tf.constant(probs_, dtype=sampler_dtype)
            num_events = tf.shape(probs)[0]
            probs_broadcast =tf.broadcast_to(tf.expand_dims(probs, axis=1), [num_events, batch_size])
            samples = Sampler._generate_bernoulli(probs=probs_broadcast,
                                                  n_sample_packs_per_probability=sample_size,
                                                  bitpack_dtype=bitpack_dtype,
                                                  dtype=sampler_dtype,
                                                  name=be_name_)
            return samples
        return _graph_op_sample_bernoulli_

    def _get_basic_event(self, basic_event_def_name: str):
        event_found, event_def = self.__lookup_item_by_name(basic_event_def_name, self._basic_event_indices, "define-basic-event")
        if event_found:
            return event_def

        #be_op = self._build_basic_event(prob_=float(event_def.value), be_name_=basic_event_def_name)

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
    def _build_op_0be_1gate(input_gate_inputs_, op_name_: Optional[str] = None):
        print(f"building op-0be-1gate for {input_gate_inputs_}")
        def _graph_op_0be_1gate_(batch_size, sample_size, sampler_dtype, bitpack_dtype):
            gate_op = input_gate_inputs_[0]
            return gate_op(batch_size, sample_size, sampler_dtype, bitpack_dtype)
        return _graph_op_0be_1gate_


    def _build_op_1be_0gate(self, be_tensor_indices_, op_name_: Optional[str] = None):
        print(f"building op-1be-0gate for {be_tensor_indices_}")
        probs_ = [self._basic_event_probs[idx] for idx in be_tensor_indices_]
        def _graph_op_1be_0gate_(batch_size, sample_size, sampler_dtype, bitpack_dtype):
            sampler_ = self._build_basic_event_subset_sampler(probs_=probs_, be_name_=op_name_)
            samples_ = sampler_(batch_size, sample_size, sampler_dtype, bitpack_dtype)
            return samples_[0, :, :]
            #be_inputs_ = tf.gather(inputs, be_tensor_indices_, name=op_name_) # tensor with shape (1, batch_size, sample_size)
            #return be_inputs_[0, :, :] # tensor with shape (batch_size, sample_size)
        return _graph_op_1be_0gate_


    def _build_op_geq_2be_0gate(self, bitwise_nary_op_, be_tensor_indices_, op_name_: Optional[str] = None):
        print(f"building op-geq-2be-0gate for {be_tensor_indices_}, op: {bitwise_nary_op_}")
        probs_ = [self._basic_event_probs[idx] for idx in be_tensor_indices_]
        def _graph_op_geq_2be_0gate_(batch_size, sample_size, sampler_dtype, bitpack_dtype):
            sampler_ = self._build_basic_event_subset_sampler(probs_=probs_, be_name_=f"be_for_{op_name_}")
            samples_ = sampler_(batch_size, sample_size, sampler_dtype, bitpack_dtype)
            return bitwise_nary_op_(inputs=samples_, name=op_name_)
            #be_inputs_ = tf.gather(inputs, be_tensor_indices_) # tensor with shape (n, batch_size, sample_size)
            #return bitwise_nary_op_(inputs=be_inputs_, name=op_name_) # tensor with shape (batch_size, sample_size)
        return _graph_op_geq_2be_0gate_

    @staticmethod
    def _build_op_0be_geq_2gate(bitwise_nary_op_, input_gate_inputs_, op_name_: Optional[str] = None):
        print(f"building op-0be-geq-2gate for {input_gate_inputs_}, op: {bitwise_nary_op_}")
        def _graph_op_0be_geq_2gate(batch_size, sample_size, sampler_dtype, bitpack_dtype):
            # Apply each function in op_inputs to inputs
            input_gate_outputs_ = [op(batch_size, sample_size, sampler_dtype, bitpack_dtype) for op in input_gate_inputs_]  # Each output has shape [batch_size, sample_size]

            # Stack outputs into a tensor of shape [num_ops, batch_size, sample_size]
            input_gate_outputs_stacked_ = tf.stack(input_gate_outputs_, axis=0)
            #input_gate_outputs_stacked_ = tf.parallel_stack(input_gate_outputs_,)

            # # Use tf.foldl to reduce over the outputs using new_op
            # result = tf.foldl(
            #     lambda accum, elem: bitwise_nary_op_(accum, elem),
            #     input_gate_outputs_stacked_[1:],  # Start from the second element if outputs_stacked has more than one element
            #     initializer=input_gate_outputs_stacked_[0]
            # )
            return bitwise_nary_op_(inputs=input_gate_outputs_stacked_, name=op_name_)
            #return bitwise_nary_op_(be_inputs_) # tensor with shape (batch_size, sample_size)
        return _graph_op_0be_geq_2gate

    def _build_op_geq_1be_geq_1gate(self, bitwise_nary_op_, be_tensor_indices_, input_gate_inputs_, op_name_: Optional[str] = None):
        print(f"building op-geq-1be-geq-1gate for BEs: {be_tensor_indices_}, Gates: {input_gate_inputs_}, op: {bitwise_nary_op_}")
        probs_ = [self._basic_event_probs[idx] for idx in be_tensor_indices_]
        def _graph_op_geq_1be_geq_1gate(batch_size, sample_size, sampler_dtype, bitpack_dtype):
            sampler_ = self._build_basic_event_subset_sampler(probs_=probs_, be_name_=f"be_for_{op_name_}")
            samples_ = sampler_(batch_size, sample_size, sampler_dtype, bitpack_dtype)
            #return bitwise_nary_op_(inputs=samples_, name=op_name_)
            # Apply each function in op_inputs to inputs
            input_gate_outputs_ = [op(batch_size, sample_size, sampler_dtype, bitpack_dtype) for op in input_gate_inputs_]  # Each output has shape [batch_size, sample_size]
            # Stack outputs into a tensor of shape [num_ops, batch_size, sample_size]
            #input_gate_outputs_stacked_ = tf.parallel_stack(input_gate_outputs_)
            input_gate_outputs_stacked_ = tf.stack(input_gate_outputs_, axis=0)
            #print("input_gate_outputs_stacked_.shape", input_gate_outputs_stacked_.shape)
            gate_inputs_ = tf.concat([samples_, input_gate_outputs_stacked_], axis=0)
            #print("gate_inputs_.shape", gate_inputs_.shape)
            # Use tf.foldl to reduce over the outputs using new_op
            # result = tf.foldl(
            #     lambda accum, elem: bitwise_nary_op_(accum, elem),
            #     gate_inputs_[1:],  # Start from the second element if outputs_stacked has more than one element
            #     initializer=gate_inputs_[0]
            # )
            return bitwise_nary_op_(inputs=gate_inputs_, name=op_name_)
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
                                                        input_gate_inputs_=gate_inputs,
                                                        op_name_=gate_def_name)
        elif num_gate_inputs >= 2 and num_be_inputs == 0:
            logic_fn_ = self._build_op_0be_geq_2gate(bitwise_nary_op_=gate_op,
                                                    input_gate_inputs_=gate_inputs,
                                                    op_name_=gate_def_name)
        elif num_gate_inputs == 0 and num_be_inputs >= 2:
            logic_fn_ = self._build_op_geq_2be_0gate(bitwise_nary_op_=gate_op,
                                                    be_tensor_indices_=be_ref_tensor_input_indices,
                                                    op_name_=gate_def_name)
        elif num_gate_inputs == 1 and num_be_inputs == 0:
            logic_fn_ = self._build_op_0be_1gate(input_gate_inputs_=gate_inputs, op_name_=gate_def_name)
        elif num_gate_inputs == 0 and num_be_inputs == 1:
            logic_fn_ = self._build_op_1be_0gate(be_tensor_indices_=be_ref_tensor_input_indices, op_name_=gate_def_name)
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

    # @contextlib.contextmanager
    # def options(opts):
    #
    #     tf.config.optimizer.set_experimental_options(opts)
    #     try:
    #         yield
    #     finally:
    #         tf.config.optimizer.set_experimental_options(old_opts)

    opts = tf.config.optimizer.get_experimental_options()
    opts['layout_optimizer'] = True
    opts['constant_folding'] = True
    opts['shape_optimization'] = True
    opts['remapping'] = True
    opts['arithmetic_optimization'] = True
    opts['dependency_optimization'] = True
    opts['loop_optimization'] = True
    opts['function_optimization'] = True
    opts['debug_stripper'] = True
    opts['disable_model_pruning'] = False
    opts['scoped_allocator_optimization'] = True
    opts['pin_to_host_optimization'] = False # True to force CPU
    opts['implementation_selector'] = True #Enable the swap of kernel implementations based on the device placement.
    opts['auto_mixed_precision'] = False #Change certain float32 ops to float16 on Volta GPUs and above. Without the use of loss scaling, this can cause numerical underflow (see keras.mixed_precision.experimental.LossScaleOptimizer).
    opts['min_graph_nodes'] = 1 #The minimum number of nodes in a graph to optimizer. For smaller graphs, optimization is skipped.
    opts['auto_parallel'] = True
    tf.config.optimizer.set_experimental_options(opts)

    #xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/demo/gas_leak.xml"
    #tf.config.run_functions_eagerly(False)
    #tf.compat.v1.disable_eager_execution()

    #xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-openpsa-models/models/Aralia/das9701.xml"
    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-openpsa-models/models/Aralia/baobab1.xml"

    #input_probs, num_inputs, logic_fn, num_outputs = opsamef_xml_to_tensorflow_module(xml_file_path)
    input_probs, num_inputs, logic_, num_outputs = OpsamefCanopyFaultTreeTranslator(xml_file_path).translate()

    print("input_probs:", input_probs)
    print("num_inputs:", num_inputs)
    print("logic_fn:", logic_)
    print("num_outputs:", num_outputs)

    import ctypes

    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128

    sampler_dtype = tf.float32
    bitpack_dtype = tf.uint8
    optimizer = BatchSampleSizeOptimizer(
        num_events=num_inputs,
        max_bytes=int(0.001 * 2 ** 32),  # 1.8 times 4 GiB
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
    sizer = optimizer.get_results()
    #
    #tf.config.run_functions_eagerly(False)
    #tf.compat.v1.disable_eager_execution()
    # concrete_logic_fn = logic_
    concrete_logic_fn = tf.function(
        func=logic_,
        jit_compile=False
    )
    #final_output = concrete_logic_fn(sizer['batch_size'], sizer['sample_size'], sampler_dtype, bitpack_dtype)
    #print(final_output)
    # (
    #     batch_size=tf.constant(value=sizer['batch_size'], dtype=tf.int32),
    # sample_size = tf.constant(value=sizer['sample_size'], dtype=tf.int32),
    # sampler_dtype = sampler_dtype,
    # bitpack_dtype = bitpack_dtype,
    # ),
    # sampler = LogicTreeBroadcastSampler(
    #     logic_fn=None,
    #     num_inputs=sizer['num_events'],
    #     num_outputs=num_outputs,
    #     num_batches=sizer['total_batches'],
    #     batch_size=sizer['batch_size'],
    #     sample_size=sizer['sample_size'],
    #     bitpack_dtype=bitpack_dtype,
    #     sampler_dtype=sampler_dtype,
    #     acc_dtype=sampler_dtype
    # )
    #sampler.tally_from_samples(efficient_bitwise_or(sampler.generate(input_probs)))
    # print(sampler.tally_from_samples(concrete_logic_fn(sampler.generate(input_probs))))
    # exit(0)
    # @tf.function(jit_compile=True)
    # def run(input_probs_):
    #     print(sampler.generate(input_probs_).shape)
    #     return logic_(sampler.generate(input_probs_))

    #print(sampler.tally(input_probs))
    # Set up the log directory
    logdir = "/home/earthperson/projects/pracciolini/logs"
    profile_dir = f"{logdir}/profiler"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    # Set up logging.
    writer = tf.summary.create_file_writer(logdir)

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=profile_dir)
    # final_output = sampler.tally_from_samples(efficient_bitwise_or(sampler.generate(input_probs)))
    # final_output2 = sampler.tally_from_samples(bitwise_or(sampler.generate(input_probs)))

    final_output = concrete_logic_fn(sizer['batch_size'], sizer['sample_size'], sampler_dtype, bitpack_dtype)

    # Run the function within the writer's context
    with writer.as_default():
        # Export the trace
        tf.summary.trace_export(
            name="FT0",
            step=0,
        )
    #tf.summary.trace_off()
    print("Final Output:")
    print(final_output)
    tf.summary.trace_off()
    exit(0)

    # count = 2
    # t = timeit.Timer(lambda: sampler.tally(input_probs))
    # times = t.timeit(number=count)
    # print(f"total_time (s): {times}")
    # print(f"avg_time (s): {times / float(count)}")
    #print(sampler.tally_from_samples(sampler.eval_fn(logic_, input_probs)))
    #rint(sampler.eval_fn(logic_, input_probs))