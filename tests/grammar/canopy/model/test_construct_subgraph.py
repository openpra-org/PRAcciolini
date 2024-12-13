import unittest

import tensorflow as tf

from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph

class SubGraphConstructionTests(unittest.TestCase):
    def setUp(self):
        # might need pop count
        samples_a = [0b00000000, 0b00000001, 0b00000010] # 8 * 3 = 24 total samples for A, P(A) = (2/24)  = 0.0833
        samples_b = [0b00000100, 0b00001000, 0b00010000] # 8 * 3 = 24 total samples for B, P(B) = (3/24)  = 0.1250
        samples_c = [0b00100100, 0b01000000, 0b10000000] # 8 * 3 = 24 total samples for C, P(C) = (4/24)  = 0.1666
        samples_d = [0b11111111, 0b11111111, 0b11111111] # 8 * 3 = 24 total samples for D, P(D) = (24/24) = 1.0000
        self.A = Tensor(tf.constant(samples_a, dtype=tf.uint8), name="A", shape=[1, None])
        self.B = Tensor(tf.constant(samples_b, dtype=tf.uint8), name="B", shape=[1, None])
        self.C = Tensor(tf.constant(samples_c, dtype=tf.uint8), name="C", shape=[1, None])
        self.D = Tensor(tf.constant(samples_d, dtype=tf.uint8), name="D", shape=[1, None])

    ## TODO::
    ## TODO:
    ##  1. we need a running average type of Expected Value Reducer, that can be fed more samples, and the estimate improves.
    ##  2. we need to do this for variance, as well as variational loss
    ##  3. we need to play with model.predict() and see what the loss metrics mean here
    ##      3.1: model.predict() has a sample_weight parameter!
    ##      3.2: model.predict() has a mask parameter!
    ##  4. we need to be able to stream data in from tf.data.DataSet, or alternatively, sample separately and stream it in

    def test_simple_ops(self):
        # f_11 = A|B|C
        # f_12 = ~D
        # f_21 = f_11 & D
        # f    = f_21 ^ f_12
        subgraph = SubGraph(name="f")
        subgraph.register_input(self.A)
        subgraph.register_input(self.B)
        subgraph.register_input(self.C)
        subgraph.register_input(self.D)
        # intermediates
        f_11 = subgraph.bitwise_or(self.A, self.B, self.C, name="f_11")
        f_21 = subgraph.bitwise_and(self.D, f_11, name="f_21")
        f_12 = subgraph.bitwise_not(self.D, name="f_12")
        # output
        f = subgraph.bitwise_xor(f_21, f_12, name="f")
        P_f = subgraph.tally(f, name="P_f")


        tf_graph = subgraph.to_tensorflow_model()
        tf_graph.save("subgraph.h5")

        # Execute the subgraph
        #feed_dict = {input_tensor: input_data}
        output_values = subgraph.execute(dict())

        # Retrieve the expected value
        expected_value = output_values[P_f].numpy()
        print(f"Expected Mean Value: {expected_value}")

        # Convert to a TensorFlow Keras model
        model = subgraph.to_tensorflow_model()

        # Use the model to predict
        predicted_value = model.predict(x=[self.A.tf_tensor, self.B.tf_tensor, self.C.tf_tensor, self.D.tf_tensor])
        print(f"Predicted Mean Value: {predicted_value}")

        return
        # run the computation
        func = four_ops_four_operands.execute_function()
        result = func()
        f_bits_computed = result[0]

        # P(F_bits_known) = (8/24) = 0.333
        f_bits_known = tf.constant(value=[0b00100100, 0b01001001, 0b10010010], name="f_bits_known", dtype=tf.uint8)

        # The result is a tuple of outputs
        self.assertTrue(
            tf.reduce_all(tf.equal(f_bits_computed, f_bits_known)),
            "computed f_bits do not match the known result"
        )

        # count all the 1-bits each element in the tensor, do this for all elements in the tensor
        pop_counts = tf.raw_ops.PopulationCount(x=f_bits_computed)

        # sum up all the counts
        total_one_bits = tf.reduce_sum(tf.cast(pop_counts, tf.uint64), axis=0)

        # Compute the total number of elements in the tensor
        total_elements = tf.size(f_bits_computed, out_type=tf.uint64)  # Returns a tensor

        words_per_element = tf.constant(value=tf.dtypes.as_dtype(f_bits_computed.dtype).size, dtype=tf.uint8) #tf.int32 by default
        bits_per_element = tf.math.multiply(words_per_element, tf.constant(value=8, dtype=tf.uint8))
        total_bits = tf.math.multiply(total_elements, tf.cast(bits_per_element, tf.uint64))

        # Compute the expected mean
        expected_mean = tf.math.divide(tf.cast(total_one_bits, tf.float64), tf.cast(total_bits, tf.float64))

        # Assert that the computed expected value matches the manual computation
        self.assertAlmostEqual((8/24), expected_mean.numpy(), places=5)


        tf_graph = four_ops_four_operands.to_tensorflow_model()
        tf_graph.save("four_ops_four_operands.h5")

    def test_expectation(self):
        subgraph = SubGraph(name="ExampleSubGraph")

        # Define input tensor
        input_data = tf.constant([[0xF0F0F0F0, 0x0F0F0F0F], [0xAAAAAAAA, 0x55555555]], dtype=tf.uint32)
        input_tensor = Tensor(tensor=input_data, name="input_tensor")

        # Register the input tensor
        subgraph.register_input(input_tensor)

        # Add the expectation operator
        output_tensor = subgraph.tally(operand=input_tensor, name="expected_value")

        # Register the output tensor
        subgraph.register_output(output_tensor)

        # Execute the subgraph
        feed_dict = {input_tensor: input_data}
        output_values = subgraph.execute(feed_dict)

        # Retrieve the expected value
        expected_value = output_values[output_tensor].numpy()
        print(f"Expected Mean Value: {expected_value}")

        # Convert to a TensorFlow Keras model
        model = subgraph.to_tensorflow_model()

        # Use the model to predict
        predicted_value = model.predict(x=[input_data.numpy()])
        print(f"Predicted Mean Value: {predicted_value}")

    def test_monte_carlo_expectation(self):
        # Create a SubGraph
        subgraph = SubGraph(name="MonteCarloExpectationTest")

        # Create input tensor with samples
        samples = tf.constant([
            [0b00100100],
            [0b01001001],
            [0b10010010],
            # You can add more samples here
        ], dtype=tf.uint8)
        input_tensor = Tensor(samples, name="input_samples")
        subgraph.register_input(input_tensor)

        # Apply expectation operator
        expected_value_tensor = subgraph.expectation(input_tensor, name="ExpectedValue")

        # Register output
        subgraph.register_output(expected_value_tensor)

        # Execute the subgraph
        func = subgraph.execute_function()

        # Execute the function
        result = func(samples)
        actual_value = result[0].numpy()

        # Compute expected value manually
        pop_counts = tf.raw_ops.PopulationCount(x=samples)
        total_bits_per_sample = tf.reduce_sum(tf.cast(pop_counts, tf.float32), axis=1)
        expected_mean = tf.reduce_mean(total_bits_per_sample).numpy()

        # Assert that the computed expected value matches the manual computation
        self.assertAlmostEqual(actual_value, expected_mean, places=5)

    def test_compare_1d_with_concat_tensors(self):
        subgraph_1d = SubGraph(name="1D")
        subgraph_1d.register_input(self.A)
        subgraph_1d.register_input(self.B)
        subgraph_1d.register_input(self.C)
        subgraph_1d.register_input(self.D)
        AorB = subgraph_1d.bitwise_or(self.A, self.B, name="1d_(A|B)")
        AorBorC =  subgraph_1d.bitwise_or(AorB, self.C, name="1d_(1d_(A|B)|C)")
        AorBorCandD =  subgraph_1d.bitwise_and(AorBorC, self.D, name="1d_(1d_(1d_(A|B)|C) & D)")
        subgraph_1d.register_output(AorBorCandD)

        results_1d = subgraph_1d.execute_function()()[0].numpy()
        binary_literals_1d = [f'0b{result:08b}' for result in results_1d]
        print(f"binary_literals_1d: {binary_literals_1d}")

        return