from collections import OrderedDict

import tensorflow as tf
from lxml.etree import Element

from pracciolini.core.decorators import translation
from pracciolini.grammar.canopy.model.subgraph import SubGraph
from pracciolini.grammar.openpsa.validate import read_openpsa_xml
from pracciolini.translator.opsamef_canopy.opsamef_to_canopy import build_definitions, \
    get_event_references_and_definitions, opsamef_ft_xml_to_canopy_subgraph



@translation('opsamef_xml', 'canopy_subgraph')
def opsamef_xml_to_canopy_subgraph(xml_data: Element) -> SubGraph:
    subgraph: SubGraph = SubGraph()
    inputs, operators, outputs = build_definitions(xml_data, subgraph)
    for operator in operators.values():
        subgraph.register_output(operator)
    subgraph.prune()
    return subgraph

@translation('opsamef_xml_file', 'canopy_subgraph')
def opsamef_xml_file_to_canopy_subgraph(file_path: str) -> SubGraph:
    xml_data = read_openpsa_xml(file_path)
    subgraph: SubGraph = opsamef_xml_to_canopy_subgraph(xml_data)
    return subgraph

@translation('opsamef_xml_file', 'canopy_subgraph_file')
def opsamef_xml_file_to_canopy_subgraph_file(file_path: str) -> str:
    output_file_path = f"translations/{file_path.split('/')[-1]}.cnpy"
    subgraph = opsamef_xml_file_to_canopy_subgraph(file_path)
    subgraph.save(output_file_path)
    return output_file_path

@translation('opsamef_xml_file', 'canopy_keras')
def opsamef_xml_file_to_canopy_keras(file_path: str) -> tf.keras.Model:
    subgraph = opsamef_xml_file_to_canopy_subgraph(file_path)
    keras_model = subgraph.to_tensorflow_model()
    return keras_model

@translation('opsamef_xml_file', 'canopy_keras_file')
def opsamef_xml_file_to_canopy_keras_file(file_path: str) -> str:
    output_file_path = f"translations/{file_path.split('/')[-1]}.h5"
    keras_model: tf.keras.Model = opsamef_xml_file_to_canopy_keras(file_path)
    keras_model.save(f"translations/{file_path.split('/')[-1]}.h5")
    return output_file_path


    # Measure elapsed time
    # start_time = time.perf_counter()  # Use time.perf_counter() for higher precision
    # results_1d = subgraph.execute_function()()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")