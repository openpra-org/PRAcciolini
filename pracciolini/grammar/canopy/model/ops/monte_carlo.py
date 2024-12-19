
import tensorflow as tf

from pracciolini.grammar.canopy.probability import monte_carlo


@tf.function(jit_compile=True)
def tally(inputs):
    return monte_carlo.expectation_with_confidence_interval(inputs)

