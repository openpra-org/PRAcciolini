from typing import Dict, Tuple, Any

from lxml.etree import Element
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Operation, Tensor, TensorShape
from tensorflow.python.framework.ops import _EagerTensorBase

from pracciolini.core.decorators import translation
from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.grammar.openpsa.validate import read_openpsa_xml
from pracciolini.translator.opsamef_canopy.sampler import pack_tensor_bits


def build_events_map(tree: Element) -> Dict[str, str]:
    events: Dict[str, str] = dict()
    event_defs_xml = tree.xpath("//define-basic-event | //define-house-event")
    for event_xml in event_defs_xml:
        event = OpsaMefXmlRegistry.instance().build(event_xml)
        events[event.name] = event.value
    return events

def build_probabilities_from_event_map(event_map:Dict[str, str], name: str = "P(x)", dtype: tf.DType = tf.float64) -> tuple[list, Operation | _EagerTensorBase]:
    cast_values = [float(value) for value in event_map.values()]
    probabilities = tf.constant(value=cast_values, name=name, dtype=dtype)
    return list(event_map.keys()), probabilities

def generate_uniform_samples(dim: Tuple | tf.TensorShape, low = 0, high = 1, seed = 372, dtype: tf.DType = tf.float64) -> tf.Tensor:
    uniform_dist = tfp.distributions.Uniform(low=tf.cast(low, dtype=dtype), high=tf.cast(high, dtype=dtype))
    uniform_samples = uniform_dist.sample(sample_shape=dim, seed=tuple([seed, seed]))
    return uniform_samples

def sample_probabilities_bit_packed(probabilities: tf.Tensor,
                                    num_samples: int = int(1e6),
                                    seed: int = int(372),
                                    sampler_dtype: tf.DType = tf.float32,
                                    bitpack_dtype:tf.DType = tf.uint8) -> (
        tuple)[Tensor, dict[str, TensorShape | int | Any]]:
    bits_per_packed_dtype = tf.dtypes.as_dtype(bitpack_dtype).size * 8
    # Calculate the number of samples rounded up to the nearest multiple of bits_per_packed_dtype
    sample_count = int(((num_samples + bits_per_packed_dtype - 1) // bits_per_packed_dtype) * bits_per_packed_dtype)
    samples_dim = (probabilities.shape[0], sample_count)
    samples = generate_uniform_samples(samples_dim, seed=seed, dtype=sampler_dtype)
    sampled_probabilities: tf.Tensor = probabilities[:, tf.newaxis] >= samples
    packed_samples = pack_tensor_bits(sampled_probabilities, dtype=bitpack_dtype)
    return packed_samples, {
        "seed": seed,
        "num_samples": sample_count,
        "bitpack_dtype": packed_samples.dtype,
        "sampler_dtype": samples.dtype,
        "sampler_shape": samples.shape
    }

@translation('opsamef_xml', 'canopy')
def opsamef_xml_to_canopy_subgraph(file_path: str) -> str:
    try:
        xml_data = read_openpsa_xml(file_path)
        events_map = build_events_map(xml_data)
        events, known_probabilities = build_probabilities_from_event_map(events_map, dtype=tf.float32)
        sampled_probabilities, other = sample_probabilities_bit_packed(known_probabilities)
        print(f"meta: {other}")
        return str(len(events_map.keys()))
    except Exception as e:
        print(f"An error occurred during translation: {e}")
    return "WIP"
