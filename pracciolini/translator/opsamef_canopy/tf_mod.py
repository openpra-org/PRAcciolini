import os
import tensorflow as tf
from lxml import etree
from typing import Dict, List, Optional

from pracciolini.grammar.canopy.module.bitwise import bitwise_and, bitwise_or, bitwise_not
from pracciolini.grammar.canopy.module.sampler import Sampler
from pracciolini.grammar.canopy.probability.monte_carlo import count_one_bits, count_bits


# Node classes to represent the fault tree structure
class Node:
    def __init__(self, name: str):
        self.name = name
        self.output: Optional[tf.Tensor] = None  # To hold the TensorFlow tensor corresponding to this node

class Gate(Node):
    def __init__(self, name: str, gate_type: str, is_top: bool = True):
        super().__init__(name=name)
        self.gate_type = gate_type
        self.children: List[Gate | BasicEvent] = []
        self.is_top = is_top

class BasicEvent(Node):
    def __init__(self, name: str, probability: float):
        super().__init__(name)
        self.probability = probability
        self.parents: List[Gate] = []

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
def generate_basic_event_samples(probabilities: List[float], batch_size: int, sample_size: int, sampler_dtype=tf.float32, bitpack_dtype=tf.uint8) -> tf.Tensor:
    # Create a tensor of probabilities with shape [num_events, batch_size, sample_size]
    probs = tf.constant(probabilities, dtype=sampler_dtype)
    num_events = tf.shape(probs)[0]
    probs_broadcast = tf.broadcast_to(tf.expand_dims(probs, axis=1), [num_events, batch_size])
    samples = Sampler._generate_bernoulli(probs=probs_broadcast,
                                          n_sample_packs_per_probability=sample_size,
                                          bitpack_dtype=bitpack_dtype,
                                          dtype=sampler_dtype)
    return samples  # Shape: [num_events, batch_size, sample_size]

# Function to apply gate operations
def apply_gate_operation(gate_type: str, inputs: List[tf.Tensor], name: str) -> tf.Tensor:
    if gate_type == "and":
        return bitwise_and(inputs, name=name)

    if gate_type == "or" or gate_type == "atleast":
        return bitwise_or(inputs, name=name)

    return bitwise_not(inputs, name=name)

# Function to build the TensorFlow computation graph
def build_tensorflow_graph(sorted_nodes: List[Node], batch_size: int, sample_size: int) -> tf.Tensor:
    # Initialize dictionaries to hold outputs
    node_outputs: Dict[str, tf.Tensor] = {}
    basic_event_probs: List[float] = []
    basic_event_indices: Dict[str, int] = {}

    # First pass: collect basic event probabilities
    for node in sorted_nodes:
        if isinstance(node, BasicEvent):
            if node.name not in basic_event_indices:
                basic_event_indices[node.name] = len(basic_event_probs)
                basic_event_probs.append(node.probability)

    # Generate samples for all basic events at once
    basic_event_samples = generate_basic_event_samples(basic_event_probs, batch_size, sample_size)
    # Map basic event outputs
    for node in sorted_nodes:
        if isinstance(node, BasicEvent):
            index = basic_event_indices[node.name]
            node.output = basic_event_samples[index]
            node_outputs[node.name] = node.output

    # Second pass: process gates
    for node in reversed(sorted_nodes):
        if isinstance(node, Gate):
            if node.output is not None:
                raise ValueError(f"Node {node.name} output is already defined!")
            child_outputs = [child.output for child in node.children]
            if any(output is None for output in child_outputs):
                raise ValueError(f"Node '{node.name}' has children with undefined outputs.")
            if len(child_outputs) == 1: # just one child, propagate forward
                node.output = child_outputs[0]
            else:
                node.output = apply_gate_operation(node.gate_type, tf.stack(child_outputs, axis=0), node.name)
            node_outputs[node.name] = node.output

    # Return the outputs of the top nodes
    top_nodes = [node for node in sorted_nodes if getattr(node, 'is_top', False)]
    if not top_nodes:
        raise ValueError("No top node found in the fault tree.")
    outputs = [node.output for node in top_nodes]
    if len(outputs) == 1:
        return outputs[0]
    else:
        # If multiple top nodes, return a tuple of outputs
        return tuple(outputs)

# Main function to tie everything together
def main():
    import argparse

    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Fault Tree Analysis using TensorFlow")
    # parser.add_argument("xml_file_path", type=str, help="Path to the fault tree XML file")
    # parser.add_argument("--batch_size", type=int, default=128, help="Batch size for sampling")
    # parser.add_argument("--sample_size", type=int, default=1024, help="Sample size for each batch")
    # args = parser.parse_args()

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

    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/demo/gas_leak.xml"
    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/demo/gas_leak.xml"
    xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/synthetic-openpsa-models/models/c7-P_0.35-0.9/ft_c7-P_0.35-0.9_100.xml"
    #xml_file_path = "/home/earthperson/projects/pracciolini/tests/fixtures/openpsa/xml/opsa-mef/generic-openpsa-models/models/Aralia/das9701.xml"

    batch_size = 64
    sample_size = 64

    # Check if the XML file exists
    if not os.path.isfile(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")

    # Parse the XML and build the node graph
    nodes_dict = parse_fault_tree(xml_file_path)
    sorted_nodes = topological_sort(nodes_dict)
    print("sorted nodes", [node.name for node in sorted_nodes])

    # Build and execute the TensorFlow graph
    @tf.function(jit_compile=False)
    def logic_function(batch_size: int, sample_size: int) -> tf.Tensor:
        outputs = build_tensorflow_graph(sorted_nodes, batch_size, sample_size)
        pop_counts = tf.raw_ops.PopulationCount(x=outputs)
        one_bits = tf.reduce_sum(input_tensor=tf.cast(x=pop_counts, dtype=tf.float64), axis=None)
        return one_bits/(tf.reduce_prod(tf.cast(outputs.shape, dtype=tf.float64)) * outputs.dtype.size * 8.0)

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
    for _ in tf.range(1):
        output = logic_function(batch_size=batch_size, sample_size=sample_size)
        print(output)

    # Run the function within the writer's context
    with writer.as_default():
        # Export the trace
        tf.summary.trace_export(
            name="FT0",
            step=0,
        )

    exit(0)
    # Process and display the results
    # For demonstration, we can compute the probability of the top event occurring
    # by calculating the mean over the batches and samples
    if isinstance(output, tuple):
        # Multiple top nodes
        for i, out in enumerate(output):
            probability = tf.reduce_mean(tf.cast(out, tf.float32)).numpy()
            print(f"Probability of top event {i}: {probability}")
    else:
        # Single top node
        probability = tf.reduce_mean(tf.cast(output, tf.float32)).numpy()
        print(f"Probability of top event: {probability}")

if __name__ == '__main__':
    main()