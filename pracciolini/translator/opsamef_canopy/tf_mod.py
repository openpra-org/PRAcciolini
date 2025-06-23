import os
from collections import defaultdict

import tensorflow as tf
from lxml import etree
from typing import Dict, List, Optional

from pracciolini.grammar.canopy.module.bitwise import bitwise_and, bitwise_or, bitwise_xor, bitwise_xnor, bitwise_nand
from pracciolini.grammar.canopy.module.sampler import Sampler


# Node classes to represent the fault tree structure
class Node:
    def __init__(self, name: str, is_top: bool = False, level: int = -1):
        self.name = name
        self.output: Optional[tf.Tensor] = None  # To hold the TensorFlow tensor corresponding to this node
        self.is_top: bool = is_top
        self.level: int = level

class Gate(Node):
    def __init__(self, name: str, gate_type: str, is_top: bool = True):
        super().__init__(name=name, is_top=is_top)
        self.gate_type = gate_type
        self.children: List[Gate | BasicEvent] = []
        self.fn_op = None

class BasicEvent(Node):
    def __init__(self, name: str, probability: float):
        super().__init__(name=name, is_top=False, level=0)
        self.probability = probability

    @property
    def children(self):
        return []



# Function to parse the XML file and build the node graph
def parse_fault_tree(xml_file_path: str) -> Dict[str, Gate | BasicEvent]:
    # Parse the XML file using lxml
    parser = etree.XMLParser(ns_clean=True, recover=True)
    xml_tree = etree.parse(xml_file_path, parser)
    root = xml_tree.getroot()

    # Build nodes and store them in a dictionary
    nodes: Dict[str, Gate | BasicEvent] = {}

    # Parse basic events from <model-data>
    for event_xml in root.xpath("//model-data/define-basic-event"):
        name = event_xml.get("name")
        float_elem = event_xml.find("float")
        probability = float(float_elem.get("value")) if float_elem is not None else 0.0
        event_node = BasicEvent(name=name, probability=probability)
        nodes[name] = event_node

    # parse all gate names and types
    for gate_xml in root.xpath("//define-fault-tree/define-gate"):
        gate_name = gate_xml.get("name")
        gate_type = str(gate_xml[0].tag)

        if gate_type == "not":
            not_gate_xml = gate_xml[0]
            child_gate_type = str(not_gate_xml[0].tag)
            match child_gate_type:
                case "and": gate_type = "nand"
                case "or": gate_type = "nor"
                case "xor": gate_type = "xnor"

        nodes[gate_name] = Gate(name=gate_name, gate_type=gate_type, is_top=True) # set all to top for now

    # Parse fault trees to find top gates
    for ft_xml in root.xpath("//define-fault-tree"):
        first_gate_xml = ft_xml.xpath(".//define-gate")[0]
        first_gate_name = first_gate_xml.get("name")
        first_gate_node: Gate = nodes[first_gate_name]
        first_gate_node.is_top = True # set the first one to true

        # Process gates defined within the fault tree
        for gate_xml in ft_xml.xpath(".//define-gate"):
            gate_name = gate_xml.get("name")
            gate_node: Gate = nodes[gate_name]
            #print(gate_node.is_top, gate_node.name, gate_node.gate_type, len(gate_node.children))
            # Determine gate type and collect children
            gate_xml_to_traverse = gate_xml
            if gate_node.gate_type == "nand" or gate_node.gate_type == "nor" or gate_node.gate_type == "xnor":
                gate_xml_to_traverse = gate_xml[0]
            elif gate_node.gate_type == "basic-event":
                print(gate_node, gate_name, gate_xml)
            elif gate_node.gate_type == "gate": # just a reference to another gate
                print(gate_node, gate_name, gate_xml)
            # There should be only one child element representing the gate type
            for elem in gate_xml_to_traverse:
                # Parse the children of the gate
                for child_elem in elem:
                    child_name = child_elem.get("name")
                    if child_name is not None:
                        child_node = nodes[child_name]
                        if isinstance(child_node, Gate):
                            child_node.is_top = False # child can't be top!
                        gate_node.children.append(child_node)
                break  # Process only the first gate type element

            # If this gate is directly under <define-fault-tree>, mark it as a top node

        break # just parse the first fault tree

    # Return the nodes dictionary
    return nodes

# Function to perform a topological sort
def topological_sort(nodes_dict: Dict[str, Gate | BasicEvent]) -> List[Gate | BasicEvent]:
    visited = set()
    sorted_nodes: List[Gate | BasicEvent] = []

    def dfs(node: Gate | BasicEvent):
        if node.name in visited:
            return
        visited.add(node.name)
        if isinstance(node, Gate):
            for child in node.children:
                dfs(child)
        sorted_nodes.append(node)

    # Identify top nodes (nodes marked as 'is_top')
    top_nodes = [node for node in nodes_dict.values() if getattr(node, 'is_top', False)]
    if not top_nodes:
        # If no top nodes are marked, treat nodes without parent as top nodes
        child_names = {child.name for node in nodes_dict.values() for child in node.children}
        top_nodes = [node for node in nodes_dict.values() if isinstance(node, Gate) and node.name not in child_names]

    # Start DFS from the top nodes
    for node in top_nodes:
        dfs(node)

    # Check for cycles (if not all nodes are visited)
    if len(visited) != len(nodes_dict):
        undefined_nodes = set(nodes_dict.keys()) - visited
        raise ValueError(f"Cycles detected or undefined nodes: {undefined_nodes}")

    return list(reversed(sorted_nodes))  # Reverse to get the correct order

# Function to generate Bernoulli samples for basic events
#@tf.function(jit_compile=True)
def generate_basic_event_samples(rng:tf.random.Generator, probabilities: List[float], batch_size: int, sample_size: int, sampler_dtype=tf.float32, bitpack_dtype=tf.uint8) -> tf.Tensor:
    # Create a tensor of probabilities with shape [num_events, batch_size, sample_size]
    probs = tf.constant(probabilities, dtype=sampler_dtype)
    num_events = tf.shape(probs)[0]
    probs_broadcast = tf.broadcast_to(tf.expand_dims(probs, axis=1), [num_events, batch_size])
    samples = Sampler._generate_bernoulli(rng=rng,
                                          probs=probs_broadcast,
                                          n_sample_packs_per_probability=sample_size,
                                          bitpack_dtype=bitpack_dtype,
                                          dtype=sampler_dtype)
    return samples  # Shape: [num_events, batch_size, sample_size]

def op_for_gate_type(gate_type: str):
    if gate_type == "and":
        return bitwise_and
    elif gate_type == "or":
        return bitwise_or
    elif gate_type == "not":
        return bitwise_or
    elif gate_type == "xor":
        return bitwise_xor
    elif gate_type == "xnor":
        return bitwise_xnor
    elif gate_type == "nand":
        return bitwise_nand
    return bitwise_or

def presort_event_nodes(sorted_nodes: List[Node]):
    # Initialize dictionaries to hold outputs
    # node_outputs: Dict[str, tf.Tensor] = {}
    basic_event_probs: List[float] = []
    basic_event_probability_indices: Dict[str, int] = {}

    sorted_basic_events: List[BasicEvent] = []
    sorted_gates: List[Gate] = []

    sorted_tops: List[Node] = []

    gates_by_level = defaultdict(list)

    # First pass: collect basic event probabilities, and set gate op type
    for node in reversed(sorted_nodes):
        if isinstance(node, BasicEvent):
            sorted_basic_events.append(node)
            if node.name not in basic_event_probability_indices:
                basic_event_probability_indices[node.name] = len(basic_event_probs)
                basic_event_probs.append(node.probability)
        elif isinstance(node, Gate):
            if len(node.children) == 0:
                print("empty node!", node.name)
            node.level = 1 + max(child.level for child in node.children)
            node.fn_op = op_for_gate_type(node.gate_type)
            sorted_gates.append(node)
            # finally, add this to node it's level
            gates_by_level[node.level].append(node)
        if node.is_top:
            sorted_tops.append(node)


    presorted_nodes = {
        "nodes": {
            "sorted": reversed(sorted_nodes),
            "tops": sorted_tops,
            "by_level": gates_by_level,
        },
        "gates": {
            "sorted": sorted_gates,
            "by_level": gates_by_level,
        },
        "basic_events": {
            "sorted": sorted_basic_events,
            "probabilities": basic_event_probs,
            "probability_indices": basic_event_probability_indices,
        },
    }

    print(f"tops: {len(sorted_tops)}, gates: {len(sorted_gates)}, basic events: {len(sorted_basic_events)}")
    print(f"total_levels:{len(gates_by_level.keys())}")
    for key, value in gates_by_level.items():
        print(f"level {key}: {len(value)}, num_inputs: {sum([len(node.children) for node in value])}, {[node.name for node in value]}")
    return presorted_nodes

#@tf.function
def build_tf_graph(presorted_nodes, rng: tf.random.Generator, batch_size: tf.int32, sample_size: tf.int32):

    basic_event_probs: List[float] = presorted_nodes["basic_events"]["probabilities"]

    # Generate samples for all basic events at once
    basic_event_samples = generate_basic_event_samples(rng=rng,
                                                       probabilities=basic_event_probs,
                                                       batch_size=batch_size,
                                                       sample_size=sample_size)

    sorted_basic_events: List[BasicEvent] = presorted_nodes["basic_events"]["sorted"]
    basic_event_probability_indices: Dict[str, int] = presorted_nodes["basic_events"]["probability_indices"]

    # Map basic event outputs
    for basic_event in sorted_basic_events:
        index = basic_event_probability_indices[basic_event.name]
        basic_event.output = basic_event_samples[index]

    # Process gates grouped by level
    level_to_gates = presorted_nodes["gates"]["by_level"]

    for level in sorted(level_to_gates.keys()):
        gates_in_level = level_to_gates[level]
        max_num_inputs = max(len(gate.children) for gate in gates_in_level)
        padded_inputs = []
        gate_functions = []
        for gate in gates_in_level:
            child_outputs = [child.output for child in gate.children]
            # Pad inputs if necessary
            if len(child_outputs) < max_num_inputs:
                padding = [tf.zeros_like(child_outputs[0])] * (max_num_inputs - len(child_outputs))
                child_outputs += padding
            # Stack inputs
            stacked_inputs = tf.stack(child_outputs, axis=0)  # Shape: [max_num_inputs, batch_size, sample_size]
            padded_inputs.append(stacked_inputs)
            gate_functions.append(gate.fn_op)
        # Stack all inputs for gates at this level
        all_inputs = tf.stack(padded_inputs, axis=0)  # Shape: [num_gates, max_num_inputs, batch_size, sample_size]
        # Process all gates
        for i in range(len(gates_in_level)):
            gate = gates_in_level[i]
            gate_inputs = all_inputs[i]
            gate_output = gate_functions[i](inputs=gate_inputs, name=gate.name)
            gate.output = gate_output

    # # apply the tensorflow bitwise op
    # sorted_gates: List[Gate] = presorted_nodes["gates"]["sorted"]
    # for gate in sorted_gates:
    #     child_outputs = [child.output for child in gate.children]
    #     gate.output = gate.fn_op(inputs=tf.stack(child_outputs, axis=0), name=gate.name)

    sorted_tops: List[Node] = presorted_nodes["nodes"]["tops"]
    outputs = tf.stack([top.output for top in sorted_tops], axis=0)
    return outputs

# Main function to tie everything together
def main():
    import argparse

    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Fault Tree Analysis using TensorFlow")
    # parser.add_argument("xml_file_path", type=str, help="Path to the fault tree XML file")
    # parser.add_argument("--batch_size", type=int, default=128, help="Batch size for sampling")
    # parser.add_argument("--sample_size", type=int, default=1024, help="Sample size for each batch")
    # args = parser.parse_args()

    #tf.config.enable_resource_variables()
    #tf.config.run_functions_eagerly(False)
    #tf.compat.v1.disable_eager_execution()
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
    #xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-pwr-openpsa-model/models/converted_SLOCA_et_Grp-1_24-02-26_16-01-54.xml"
    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/synthetic-openpsa-models/models/c1-P_0.01-0.05/ft_c1-P_0.01-0.05_5000.xml"
    #xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-openpsa-models/models/Aralia/edfpa14b.xml"

    GLOBAL_RNG = tf.random.Generator.from_non_deterministic_state()
    GLOBAL_BATCH_SIZE = 2
    GLOBAL_SAMPLE_SIZE = 2**10

    # Check if the XML file exists
    if not os.path.isfile(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")

    # Parse the XML and build the node graph
    nodes_dict = parse_fault_tree(xml_file_path)
    sorted_nodes = topological_sort(nodes_dict)
    presorted_nodes = presort_event_nodes(sorted_nodes=sorted_nodes)

    @tf.function(jit_compile=True)
    def logic_function(batch_size: int, sample_size: int) -> tf.Tensor:
        graph_outputs = build_tf_graph(presorted_nodes=presorted_nodes, rng=GLOBAL_RNG, batch_size=batch_size, sample_size=sample_size)
        pop_counts = tf.raw_ops.PopulationCount(x=graph_outputs)
        one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=tf.float64), axis=None)
        return one_bits/(tf.reduce_prod(tf.cast(graph_outputs.shape, dtype=tf.float64)) * graph_outputs.dtype.size * 8.0)

    for _ in tf.range(10):
        output_ = logic_function(batch_size=GLOBAL_BATCH_SIZE, sample_size=GLOBAL_SAMPLE_SIZE)
        print(output_)
    exit(0)

    logdir = "/home/earthperson/projects/pracciolini/logs"
    profile_dir = f"{logdir}/profiler"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    # Set up logging.
    writer = tf.summary.create_file_writer(logdir)

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=profile_dir)
    # final_output = sampler.tally_from_samples(efficient_bitwise_or(sampler.generate(input_probs)))
    # final_output2 = sampler.tally_from_samples(bitwise_or(sampler.generate(input_probs)))

    # Execute the function
    for _ in tf.range(10):
        output = logic_function(batch_size=batch_size, sample_size=sample_size)
        print(output)

    # Run the function within the writer's context
    with writer.as_default():
        # Export the trace
        tf.summary.trace_export(
            name="FT0",
            step=0,
        )

if __name__ == '__main__':
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    TF_XLA_FLAGS = "--tf_xla_enable_lazy_compilation=false --tf_xla_auto_jit=2"
    TF_XLA_FLAGS += " --tf_mlir_enable_mlir_bridge=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_multiple_local_cpu_devices=true"
    TF_XLA_FLAGS += " --tf_xla_deterministic_cluster_names=true --tf_xla_disable_strict_signature_checks=true"
    TF_XLA_FLAGS += " --tf_xla_persistent_cache_directory='./xla/cache/' --tf_xla_persistent_cache_read_only=false"

    os.environ["TF_XLA_FLAGS"] = TF_XLA_FLAGS

    # TF_XLA_FLAGS="--tf_xla_always_defer_compilation=true"
    # TF_XLA_FLAGS = "--tf_xla_enable_lazy_compilation=false"
    # # TF_XLA_FLAGS="$TF_XLA_FLAGS --tf_xla_enable_lazy_compilation=true --tf_xla_auto_jit=2  --tf_xla_compile_on_demand=true"
    # TF_XLA_FLAGS = "$TF_XLA_FLAGS --tf_xla_print_cluster_outputs=true"
    # TF_XLA_FLAGS = "$TF_XLA_FLAGS --tf_mlir_enable_mlir_bridge=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_multiple_local_cpu_devices=true"
    # TF_XLA_FLAGS = "$TF_XLA_FLAGS --tf_xla_deterministic_cluster_names=true --tf_xla_disable_strict_signature_checks=true"
    # TF_XLA_FLAGS = "$TF_XLA_FLAGS --tf_xla_persistent_cache_directory='./xla/cache/' --tf_xla_persistent_cache_read_only=false"

    main()