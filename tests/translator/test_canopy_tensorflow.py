import unittest
import numpy as np
import tensorflow as tf

from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph

class TestTranslateCanopyToTensorflow(unittest.TestCase):
    def test_to_tensorflow_graph(self):
        """
        Test the `to_tensorflow_graph` method of the SubGraph class by creating a simple
        SubGraph with bitwise operations and verifying that the TensorFlow graph execution
        matches the expected output.
        """
        # Disable eager execution
        tf.compat.v1.disable_eager_execution()

        # Create input data
        input_data_a = np.array([1, 2, 3], dtype=np.uint32)
        input_data_b = np.array([4, 5, 6], dtype=np.uint32)

        tensor_a = Tensor(tf.constant(input_data_a, dtype=tf.uint32), name="A")
        tensor_b = Tensor(tf.constant(input_data_b, dtype=tf.uint32), name="B")

        # Create SubGraph and register input tensors
        subgraph = SubGraph(name="TestGraph")
        subgraph.register_input(tensor_a)
        subgraph.register_input(tensor_b)

        # Add a bitwise AND operation
        tensor_c = subgraph.bitwise_and(tensor_a, tensor_b, name="C")

        # Register output tensor
        subgraph.register_output(tensor_c)

        # Convert SubGraph to TensorFlow graph
        tf_graph = subgraph.to_tensorflow_graph()

        # Prepare to execute the TensorFlow graph
        with tf.compat.v1.Session(graph=tf_graph) as sess:
            # Initialize variables if there are any (not needed here since we use placeholders)
            # sess.run(tf.compat.v1.global_variables_initializer())

            # Get the input placeholders and output tensors by name
            a_placeholder = tf_graph.get_tensor_by_name("A:0")
            b_placeholder = tf_graph.get_tensor_by_name("B:0")
            output_c = tf_graph.get_tensor_by_name("output_0:0")  # Output tensor named 'output_0'

            # Run the graph with the input data
            result = sess.run(output_c, feed_dict={
                a_placeholder: input_data_a,
                b_placeholder: input_data_b,
            })

            # Expected result using NumPy bitwise AND operation
            expected_result = np.bitwise_and(input_data_a, input_data_b)

            # Check if the result matches the expected result
            np.testing.assert_array_equal(result, expected_result)

            print(f"Input A: {input_data_a}")
            print(f"Input B: {input_data_b}")
            print(f"Result of bitwise AND operation: {result}")
            print(f"Expected result: {expected_result}")

if __name__ == '__main__':
    unittest.main()
