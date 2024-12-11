from typing import Tuple, Any, List
from collections import OrderedDict

from lxml.etree import Element
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Operation
from tensorflow.python.framework.ops import _EagerTensorBase

from pracciolini.core.decorators import translation
from pracciolini.grammar.canopy.io.OpCode import OpCode
from pracciolini.grammar.canopy.model.subgraph import SubGraph
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.validate import read_openpsa_xml
from pracciolini.grammar.openpsa.xml.expression.logical import NotExpression
from pracciolini.translator.opsamef_canopy.sampler import pack_tensor_bits


def build_events_map(tree: Element) -> OrderedDict[str, str]:
    """
    Constructs a mapping of event names to their corresponding values from an XML tree.

    This function parses the provided XML tree to extract basic and house event definitions.
    It utilizes the OpsaMefXmlRegistry to build event objects and maps each event's name to its value.

    Args:
        tree (Element): The root element of the XML tree containing event definitions.

    Returns:
        Dict[str, str]: A dictionary where keys are event names and values are their corresponding values.
    """
    events: OrderedDict[str, str] = OrderedDict()
    event_defs_xml = tree.xpath("//define-basic-event | //define-house-event")
    for event_xml in event_defs_xml:
        event = OpsaMefXmlRegistry.instance().build(event_xml)
        events[event.name] = event.value
    return events


def build_probabilities_from_event_map(
    event_map: OrderedDict[str, str],
    name: str = "X",
    dtype: tf.DType = tf.float64
) -> Tuple[list, Operation | _EagerTensorBase]:
    """
    Generates a list of event names and a TensorFlow constant tensor representing their probabilities.

    This function converts the event values from the event map to floating-point numbers and
    creates a TensorFlow constant tensor with the specified name and data type.

    Args:
        event_map (Dict[str, str]): A mapping of event names to their string values.
        name (str, optional): The name assigned to the TensorFlow constant tensor. Defaults to "P(x)".
        dtype (tf.DType, optional): The data type of the TensorFlow constant. Defaults to tf.float64.

    Returns:
        Tuple[list, Operation | _EagerTensorBase]:
            - A list of event names.
            - A TensorFlow constant tensor containing the probability values.
    """
    cast_values = [float(value) for value in event_map.values()]
    probabilities = tf.constant(value=cast_values, name=name, dtype=dtype)
    return list(event_map.keys()), probabilities


def generate_uniform_samples(
    dim: Tuple | tf.TensorShape,
    low: float = 0,
    high: float = 1,
    seed: int = 372,
    dtype: tf.DType = tf.float64
) -> tf.Tensor:
    """
    Generates uniformly distributed samples within a specified range and shape.

    This function creates a uniform distribution using TensorFlow Probability and samples
    values based on the provided dimensions, seed, and data type.

    Args:
        dim (Tuple | tf.TensorShape): The shape of the desired sample tensor.
        low (float, optional): The lower bound of the uniform distribution. Defaults to 0.
        high (float, optional): The upper bound of the uniform distribution. Defaults to 1.
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 372.
        dtype (tf.DType, optional): The data type of the generated samples. Defaults to tf.float64.

    Returns:
        tf.Tensor: A tensor containing uniformly distributed samples.
    """
    uniform_dist = tfp.distributions.Uniform(low=tf.cast(low, dtype=dtype), high=tf.cast(high, dtype=dtype))
    uniform_samples = uniform_dist.sample(sample_shape=dim, seed=seed)
    return uniform_samples


def sample_probabilities_bit_packed(
    probabilities: tf.Tensor,
    num_samples: int = int(2 ** 20),
    seed: int = 372,
    sampler_dtype: tf.DType = tf.float32,
    bitpack_dtype: tf.DType = tf.uint8
) -> Tuple[tf.Tensor, OrderedDict[str, Any]]:
    """
    Samples probabilities and packs the sampled bits into a compact format.

    This function generates uniform samples, compares them against the provided probabilities
    to determine sampled events, and then packs the resulting boolean tensor into bit-packed format.

    Args:
        probabilities (tf.Tensor): A tensor of probabilities for each event.
        num_samples (int, optional): The number of samples to generate. Defaults to int(1e6).
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 372.
        sampler_dtype (tf.DType, optional): Data type for the sampler tensor. Defaults to tf.float32.
        bitpack_dtype (tf.DType, optional): Data type for the bit-packed samples. Defaults to tf.uint8.

    Returns:
        Tuple[Tensor, Dict[str, Any]]:
            - A tensor containing the bit-packed sampled probabilities.
            - A dictionary with metadata about the sampling process, including seed, number of samples,
              data types, and sampler shape.
    """
    bits_per_packed_dtype = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    # Calculate the number of samples rounded up to the nearest multiple of bits_per_packed_dtype
    sample_count = int(((num_samples + bits_per_packed_dtype - 1) // bits_per_packed_dtype) * bits_per_packed_dtype)
    samples_dim = (probabilities.shape[0], sample_count)
    samples = generate_uniform_samples(samples_dim, seed=seed, dtype=sampler_dtype)
    sampled_probabilities: tf.Tensor = probabilities[:, tf.newaxis] >= samples
    packed_samples = pack_tensor_bits(sampled_probabilities, dtype=bitpack_dtype)
    return packed_samples, OrderedDict({
        "seed": seed,
        "num_samples": sample_count,
        "bitpack_dtype": packed_samples.dtype,
        "sampler_dtype": samples.dtype,
        "sampler_shape": samples.shape
    })

def register_input_events(graph: SubGraph, stacked_tensor:tf.Tensor, events_map: OrderedDict[str, str], axis=0) -> OrderedDict[str, Tensor]:
    unstacked = tf.unstack(stacked_tensor, axis=axis)
    input_tensor_map: OrderedDict[str, Tensor] = OrderedDict()
    for tensor_slice, event_name in zip(unstacked, events_map.keys()):
        tensor = Tensor(tensor=tf.constant(value=tensor_slice, name=event_name), name=event_name)
        graph.register_input(tensor)
        input_tensor_map[event_name] = tensor
    return input_tensor_map

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

def build_operations_map(tree: Element, subgraph: SubGraph, input_tensors_map: OrderedDict[str, Tensor]):
    ops: OrderedDict[str, Tensor] = OrderedDict()
    level_1_gates = tree.xpath("//define-gate[not(./*/*[not(self::basic-event or self::house-event)])]")
    for gate_def_xml in level_1_gates:
        op_name, input_tensors_map = build_operation(gate_def_xml, subgraph, input_tensors_map)
        if op_name is not None:
            ops[op_name] = input_tensors_map[op_name]
    return ops, input_tensors_map

def bit_pack_samples(prob, num_samples = int(2 ** 10), seed=372, sampler_dtype=tf.float32, bitpack_dtype=tf.uint8):
    bits_per_packed_dtype = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    sample_count = int(((num_samples + bits_per_packed_dtype - 1) // bits_per_packed_dtype) * bits_per_packed_dtype)
    samples_dim = (1, sample_count)
    samples = generate_uniform_samples(samples_dim, seed=seed, dtype=sampler_dtype)
    probability = tf.constant(value=prob, dtype=sampler_dtype)
    sampled_probabilities: tf.Tensor = probability >= samples
    packed_samples = pack_tensor_bits(sampled_probabilities, dtype=bitpack_dtype)
    return packed_samples

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

def build_definitions(tree: Element, subgraph: SubGraph):
    inputs = OrderedDict()
    operators = OrderedDict()
    outputs = OrderedDict()

    referenced_events = get_referenced_events(tree)
    event_definitions = get_event_definitions(tree)
    referenced_but_not_defined = referenced_events - event_definitions
    print(f"\nundefined references: {referenced_but_not_defined}")
    defined_but_not_referenced = event_definitions - referenced_events
    print(f"unreferenced definitions: {defined_but_not_referenced}")
    unvisited_event_definitions = event_definitions.copy()
    print(f"unvisited definitions: {unvisited_event_definitions}")

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



## note: we are bit-packing the samples, which is fine, but it only gives us linear speed up
## rather, we should be using the bit-fields as gate inputs. for example, if we see that an event tree has 12 functional
# events, we should use a 16-bit dtype to encode the state of each of the functional events (what we have been doing
# with PLA).

@translation('opsamef_xml', 'canopy_subgraph')
def opsamef_xml_to_canopy_subgraph(file_path: str) -> SubGraph:
    xml_data = read_openpsa_xml(file_path)
    subgraph: SubGraph = SubGraph()
    inputs, operators, outputs = build_definitions(xml_data, subgraph)
    for operator in operators.values():
        subgraph.register_output(operator)
    tf_graph = subgraph.to_tensorflow_model()
    tf_graph.save(f"{file_path.split('/')[-1]}.h5")

    # Measure elapsed time
    # start_time = time.perf_counter()  # Use time.perf_counter() for higher precision
    # results_1d = subgraph.execute_function()()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return subgraph