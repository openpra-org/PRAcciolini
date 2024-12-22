import unittest
import timeit

from tensorflow.python.framework.dtypes import int4_ref

from pracciolini.grammar.canopy.model.layers import BitpackedBernoulli, Expectation
from pracciolini.grammar.canopy.model.ops.bitwise import bitwise_xor, bitwise_or
from pracciolini.grammar.canopy.model.ops.monte_carlo import tally
from pracciolini.grammar.canopy.model.ops.sampler import generate_bernoulli
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph

import tensorflow as tf
import numpy as np

from pracciolini.grammar.canopy.probability import monte_carlo
from pracciolini.grammar.canopy.sampler.sizer import BatchSampleSizeOptimizer


def create_batched_streaming_dataset(probs, num_samples, dtype=tf.float32):
    # Create a dataset that generates individual samples
    def generator():
        for _ in range(num_samples):
            yield probs

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(len(probs),), dtype=dtype)
    )
    return dataset

def create_streaming_dataset(probs, num_batches, batch_size, dtype=tf.float32):
    # Create a dataset that generates batches of probabilities and counts
    def generator():
        for _ in range(num_batches):
            yield (
                np.tile(probs, (batch_size, 1)),
            )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, len(probs)), dtype=dtype),
        )
    )
    return dataset


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

    def test_input_estimator_batched(self):
        subgraph = SubGraph("direct_sampler")
        input_dtype = tf.uint8
        input_tensor_spec = tf.constant(value=0, shape=(), dtype=input_dtype)
        print(input_tensor_spec.shape)
        input_x = Tensor(input_tensor_spec, name="input_x")
        subgraph.register_input(input_x)
        not_x = subgraph.bitwise_not(input_x, name="not_x")
        not_not_x = subgraph.bitwise_not(not_x, name="not_not_x")
        subgraph.register_output(not_not_x)
        output_x = subgraph.tally(not_not_x)

        model: tf.keras.Model = subgraph.to_tensorflow_model()
        print(model.summary())

        known_x = 0.1
        total_samples = 2 ** 28
        batch_size = 2 ** 24
        num_batches = int(total_samples / batch_size)
        x = DEPRECATED_Bernoulli(probs=known_x, pack_bits_dtype=input_dtype)

        # Create a dataset of samples
        def generate_samples():
            for batch_num in range(num_batches):
                samples_x = x.sample((x.bitpack_bits_per_dtype, batch_size), seed=372 + batch_num)
                yield samples_x

        dataset = tf.data.Dataset.from_generator(
            generate_samples,
            output_signature=tf.TensorSpec(shape=(batch_size,), dtype=input_dtype)
        )

        # Process the dataset
        losses = []
        batch_indices = []


        for batch_num, samples_x in enumerate(dataset):
            predicted_value = model.predict(x=samples_x, batch_size=batch_size)
            predicted_mean = predicted_value[1]
            loss = (predicted_mean - known_x) ** 2
            losses.append(loss)
            batch_indices.append(batch_num)
            print(f"Batch {batch_num}, Predicted Mean: {predicted_mean}, Loss: {loss}")

    def test_input_estimator_batched_basic_expr(self):
        # Create sample_shape and seed inputs
        sample_shape_tensor = tf.keras.Input(shape=(1,), dtype=tf.uint64)
        stub_input_data = tf.constant(value=(1, 1), dtype=np.int32)

        num_events = 128
        batch_size = 2 ** 24
        dtype = tf.uint8
        sampler_dtype = tf.float64
        probabilities = [float(1.0)/float(x+1) for x in range(num_events)]
        samples = BitpackedBernoulli(name=f"p_samples",
                                     probs=probabilities,
                                     batch_size=batch_size,
                                     dtype=dtype,
                                     sampler_dtype=sampler_dtype)(sample_shape_tensor)

        subgraph = SubGraph(name="F")
        basic_events = subgraph.register_input(Tensor(samples, name="samples"))
        input_dtype = tf.uint8
        bitpack_bits_per_dtype = 8
        ## todo:: change the TEnsor constructor to  accept just a tf.tensorspec
        input_tensor_spec = tf.constant(value=0, shape=(), dtype=input_dtype)
        subgraph = SubGraph(name="F")
        a = subgraph.register_input(Tensor(input_tensor_spec, name="a"))
        b = subgraph.register_input(Tensor(input_tensor_spec, name="b"))
        c = subgraph.register_input(Tensor(input_tensor_spec, name="c"))
        d = subgraph.register_input(Tensor(input_tensor_spec, name="d"))
        e = subgraph.register_input(Tensor(input_tensor_spec, name="e"))
        f = subgraph.register_input(Tensor(input_tensor_spec, name="f"))

        # intermediates
        f_11 = subgraph.bitwise_and(a, b, c, d, name="f_11")
        f_13 = subgraph.bitwise_xor(e, f, name="f_13")
        f_12 = subgraph.bitwise_not(d, name="f_12")
        f_21 = subgraph.bitwise_and(f_12, f_11, name="f_21")
        # output
        f = subgraph.bitwise_or(f_21, f_13, name="F")
        P_f = subgraph.tally(f, name="P_F")

        model: tf.keras.Model = subgraph.to_tensorflow_model()
        print(model.summary())

        model.save("six_operands.h5")
        known_x = 0.1
        total_samples = 2 ** 27
        batch_size = 2 ** 24
        num_batches = int(total_samples / batch_size)

        P_a = DEPRECATED_Bernoulli(probs=0.01, pack_bits_dtype=input_dtype)
        P_b = DEPRECATED_Bernoulli(probs=0.02, pack_bits_dtype=input_dtype)
        P_c = DEPRECATED_Bernoulli(probs=0.03, pack_bits_dtype=input_dtype)
        P_d = DEPRECATED_Bernoulli(probs=0.04, pack_bits_dtype=input_dtype)
        P_e = DEPRECATED_Bernoulli(probs=0.001, pack_bits_dtype=input_dtype)
        P_f = DEPRECATED_Bernoulli(probs=0.002, pack_bits_dtype=input_dtype)

        # Create a dataset of samples
        def generate_samples():
            for batch_num in range(num_batches):
                samples_a = P_a.sample((bitpack_bits_per_dtype, batch_size))
                samples_b = P_b.sample((bitpack_bits_per_dtype, batch_size))
                samples_c = P_c.sample((bitpack_bits_per_dtype, batch_size))
                samples_d = P_d.sample((bitpack_bits_per_dtype, batch_size))
                samples_e = P_e.sample((bitpack_bits_per_dtype, batch_size))
                samples_f = P_f.sample((bitpack_bits_per_dtype, batch_size))
                yield samples_a, samples_b, samples_c, samples_d, samples_e, samples_f

        dataset = tf.data.Dataset.from_generator(
            generate_samples,
            output_signature=(
                tf.TensorSpec(shape=(batch_size,), dtype=input_dtype),
                tf.TensorSpec(shape=(batch_size,), dtype=input_dtype),
                tf.TensorSpec(shape=(batch_size,), dtype=input_dtype),
                tf.TensorSpec(shape=(batch_size,), dtype=input_dtype),
                tf.TensorSpec(shape=(batch_size,), dtype=input_dtype),
                tf.TensorSpec(shape=(batch_size,), dtype=input_dtype),
            )
        )

        # Process the dataset
        losses = []
        batch_indices = []

        cumulative_sum = 0.0
        total_samples_processed = 0

        for batch_num, (samples_a, samples_b, samples_c, samples_d, samples_e, samples_f) in enumerate(dataset):
            predicted_value = model.predict(
                x=[samples_a, samples_b, samples_c, samples_d, samples_e, samples_f],
                batch_size=batch_size
            )

            # Assuming predicted_value is a numpy array or can be converted to one
            batch_sum = predicted_value
            #batch_size = predicted_value.shape[0]  # Number of samples in the batch

            # Update cumulative sum and total samples
            cumulative_sum += batch_sum
            total_samples_processed += 1

            # Compute cumulative mean
            cumulative_mean = cumulative_sum / total_samples_processed

            # Compute loss based on cumulative mean
            loss = (cumulative_mean - known_x) ** 2
            losses.append(loss)
            batch_indices.append(batch_num)

            print(f"Batch {batch_num}, predicted {predicted_value} Cumulative Mean: {cumulative_mean}, Loss: {loss}")

    def test_concat_input_estimator_batched_basic_expr(self):

        # Create sample_shape and seed inputs
        batch_size = 2 ** 16
        stub_input_data = tf.constant(value=batch_size, dtype=tf.int32)
        sample_shape_tensor = tf.keras.Input(shape=(), dtype=tf.int32)
        num_events = 32
        iterations = 2

        dtype = tf.uint8
        sampler_dtype = tf.float64
        samples_per_batch = tf.dtypes.as_dtype(dtype).size * 8
        probabilities = [float(1.0)/float(x+1) for x in range(num_events)]
        total_samples = num_events * batch_size * samples_per_batch * iterations
        print(f"total samples: 2^{np.log(total_samples)/np.log(2)}")
        print(f"samples per iteration: 2^{np.log(total_samples/iterations)/np.log(2)}")
        print(f"samples per iteration per event: 2^{np.log(total_samples/iterations/num_events)/np.log(2)}")
        Sampler = BitpackedBernoulli(name=f"p_samples",
                                     probs=probabilities,
                                     batch_size=batch_size,
                                     dtype=dtype,
                                     sampler_dtype=sampler_dtype)

        samples = Sampler(sample_shape_tensor)
        # op_nand = BitwiseNand()(samples[:, 0:5])
        # op_xor = BitwiseXor()(samples[:, 3:8])
        # op_or = BitwiseOr()(Concatenate(dtype=tf.uint8, axis=1)([op_nand, samples[:, 1:2], op_xor]))
        # op_not = BitwiseNot()(op_or)
        #outputs = [Expectation()(op_not), Expectation()(samples)]
        outputs = [Expectation()(samples)]
        model = tf.keras.Model(inputs=sample_shape_tensor, outputs=outputs)
        model.compile()
        model.summary()
        log_dir = "~/projects/pracciolini/logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.save("16_events_4_gates_20_expectations.h5")

        print(probabilities)
        for i in range(iterations):
            outputs = model.predict(x=tf.reshape(stub_input_data, (1,)), callbacks=[tensorboard_callback])
            outputs = tf.transpose(outputs)
            print(f"5th, mean, 95th")
            print((outputs.numpy()))


    def test_input_estimator(self):
        #num_samples
        batch_size = 64
        num_batches = 30
        bitpacked_samples_tensorspec = bit_pack_samples(
                            prob=0.5,
                            num_samples=int(batch_size),
                            seed=372,
                            sampler_dtype=tf.float64,
                            bitpack_dtype=tf.uint8)

        input_x = Tensor(bitpacked_samples_tensorspec, name="input_x")
        subgraph = SubGraph("direct_sampler")
        subgraph.register_input(input_x)
        not_x = subgraph.bitwise_not(input_x, name="not_x")
        not_not_x = subgraph.bitwise_not(not_x, name="not_not_x")
        subgraph.register_output(not_not_x)
        output_x = subgraph.tally(not_not_x)

        model: tf.keras.Model = subgraph.to_tensorflow_model()
        print(model.summary())

        model.compile(loss="mse")
        sample_shape = (num_batches, batch_size)
        x = DEPRECATED_Bernoulli(probs=0.5, dtype=tf.bool)
        samples_x = x.sample(sample_shape, seed=372, pack_bits=tf.uint8)
        samples_x = tf.reshape(samples_x, [-1, 8])
        print(samples_x.shape)
        predicted_value = model.predict(x=samples_x)
        print(f"Predicted Mean Value: {predicted_value}")







            #model.evaluate(x=sample)

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

        print(model.summary())

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
            [0b11110000],
            [0b01010101],
            [0b10000111],
            # You can add more samples here
        ], dtype=tf.uint8)
        input_tensor = Tensor(samples, name="input_samples")
        subgraph.register_input(input_tensor)

        # Apply expectation operator
        expected_value_tensor = subgraph.tally(input_tensor, name="ExpectedValue")

        # Execute the subgraph
        func = subgraph.execute_function()

        # Execute the function
        results = func(samples)
        subgraph_computed_value = results[0].numpy()
        tf_keras_computed_value = subgraph.to_tensorflow_model().predict(x=[samples])

        # Assert that the computed expected value matches the manual computation
        self.assertAlmostEqual(subgraph_computed_value, 0.5, places=5)
        self.assertAlmostEqual(tf_keras_computed_value, 0.5, places=5)

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

    def test_parametrized_seed_and_batch_v0(self):

        batch_size = 9
        stub_input_data = tf.constant(value=batch_size, dtype=tf.int32)
        sample_shape_tensor = tf.keras.Input(shape=(), dtype=tf.int32)
        width = 7
        iterations = 3
        probabilities = [float(1.0)/float(x+1) for x in range(width)]

        probs = tf.constant(probabilities, dtype=tf.float64)
        samples: tf.Tensor = generate_bernoulli(probs=probs, count=batch_size, bitpack_dtype=tf.uint8, dtype=tf.float64, seed=1234)

        # Use the function in the Lambda layer
        samples_layer = tf.keras.layers.Lambda(generate_bernoulli)(sample_shape_tensor)
        outputs = [samples_layer]
        model = tf.keras.Model(inputs=sample_shape_tensor, outputs=outputs)
        model.compile()
        model.summary()
        for i in range(iterations):
            outputs = model.predict(x=stub_input_data)
            print((outputs.numpy()))


    def test_generate_bernoulli(self):
        #tf.profiler.experimental.start('../../../../logs')
        #tf.compat.v1.disable_eager_execution()
        tf.config.run_functions_eagerly(False)

        sampler_dtype = tf.float32
        bitpack_dtype = tf.uint8
        num_events = 2 ** 10
        probs = tf.constant([
            [1.0 / (2.0) for x in range(num_events)],
        ], dtype=sampler_dtype)
        optimizer = BatchSampleSizeOptimizer(
            num_events=num_events,
            max_bytes=int(1.8 * 2 ** 32),  # 1.8 times 4 GiB
            sampled_bits_per_event_range=(None, None),
            sampler_dtype=sampler_dtype,
            bitpack_dtype=bitpack_dtype,
            batch_size_range=(1, None),
            sample_size_range=(1, None),
            total_batches_range=(1, None),
            max_iterations=3000,
            tolerance=1e-8,
        )
        optimizer.optimize()
        sample_sizer = optimizer.get_results()

        losses, est_mean = monte_carlo.batched_estimate(probs_=tf.broadcast_to(probs, [sample_sizer['batch_size'], num_events]),
                         num_batches_=sample_sizer['total_batches'],
                         sample_size_=sample_sizer['sample_size'],
                         bitpack_dtype_=bitpack_dtype,
                         sampler_dtype_=sampler_dtype)


        print(f"Batch Losses: {losses}")
        print(f"Known Probabilities ({len(probs.numpy()[0])}): {probs.numpy()}")
        print(f"Estimated Means    ({len(est_mean.numpy())}): {est_mean.numpy()}")


    def test_generate_bernoulli_and_eval_gates(self):
        # tf.profiler.experimental.start('../../../../logs')
        # tf.compat.v1.disable_eager_execution()
        tf.config.run_functions_eagerly(False)

        @tf.function(jit_compile=True)
        ## P(A) = 0.5
        ## P(D) = 0.2
        ## P(E) = 0.16667
        ## P(outputs) = 0.5 = P[((A | D) ^ E) & (A | D) = (A | D) & ~E]
        def some_logic_expression(inputs):
            # input_shape = [batch_size, num_events, sample_size]
            #print(inputs.shape)
            g1 = tf.bitwise.bitwise_or(inputs[:, 0, :], inputs[:, 3, :])
            g2 = tf.bitwise.bitwise_xor(inputs[:, 4, :], g1)
            g3 = tf.bitwise.bitwise_and(g1, g2)
            outputs = g3
            # output_shape = [batch_size, sample_size] (for a single output)
            # to be
            return outputs

        #@tf.function(jit_compile=True)
        ## P(A) = 0.5
        ## P(D) = 0.2
        ## P(E) = 0.16667
        ## P(X) = 0.000001
        ## P(F) = probability of failure; P(outputs) = 0.5 = P[((A | D) ^ E) & (A | D) = (A | D) & ~E]


        "F = ((AC | DC) ^ E) & (A | D) millions in size -> expansion 2^(n) -> contraction --> (A | D) & ~E"
        "P(F) = P[(A|D)&~E)] = [P(A) + P(D) - P(A)•P(D)] • [1 - P(E)]"
        "P(A|B|C) =~ P(A) + P(B) + P(C)  + P(A)P(B) + P(A)P(C) + P(B)P(C)"  # 2^(N)

        "P(F) = P(A|B); E[A | B] -> P(F) - O(N)"

        def another_logic_expression(inputs):
            return bitwise_or(inputs)

        @tf.function(jit_compile=True)
        def batched_estimate_outputs(
                             input_probs_: tf.Tensor,
                             output_probs_: tf.Tensor,
                             fn_logic,
                             num_batches_: int,
                             sample_size_: int,
                             bitpack_dtype_: tf.DType,
                             sampler_dtype_: tf.DType,
                             acc_dtype_: tf.DType = tf.uint64):
            # Get dynamic shapes
            batch_size_ = tf.shape(input_probs_)[0] # batch_size_ remains the same for inputs and outputs

            #num_input_events_ = tf.shape(input_probs_)[1]
            #num_output_events_ = tf.shape(output_probs_)[1]

            # Convert sample_size_ and num_batches_ to tensors if needed
            num_batches_tensor = tf.constant(num_batches_, dtype=tf.int32)
            sample_size_tensor = tf.constant(sample_size_, dtype=acc_dtype_)

            # Get bitpack size in bytes as tensor
            bitpack_size_bytes = tf.constant(tf.dtypes.as_dtype(bitpack_dtype_).size, dtype=acc_dtype_)

            # Compute event dimension size
            event_dim_size_ = sample_size_tensor * bitpack_size_bytes * tf.constant(8, dtype=acc_dtype_)
            event_bits_in_batch_ = event_dim_size_

            # Initialize tensors
            cumulative_one_bits_ = tf.zeros((batch_size_,), dtype=acc_dtype_)
            losses_ = tf.TensorArray(dtype=tf.float64, size=num_batches_tensor)

            for batch_idx_ in tf.range(num_batches_tensor):
                # Generate samples
                packed_bits_ = generate_bernoulli(
                    probs=input_probs_,
                    n_sample_packs_per_probability=sample_size_,
                    bitpack_dtype=bitpack_dtype_,
                    dtype=sampler_dtype_,
                )
                # print(packed_bits_.shape)
                output_packed_bits_ = fn_logic(packed_bits_)
                #print(output_packed_bits_.shape)

                # Compute the number of one-bits
                one_bits_in_batch_ = tf.reduce_sum(
                    tf.cast(
                        tf.raw_ops.PopulationCount(x=output_packed_bits_),
                        dtype=acc_dtype_
                    ),
                    axis=-1
                )
                # print(f"output_packed_bits_.shape: {output_packed_bits_.shape}")
                # print(f"one_bits_in_batch_.shape: {one_bits_in_batch_.shape}")
                # Update cumulative counts
                cumulative_one_bits_ += one_bits_in_batch_
                cumulative_bits_ = (tf.cast(batch_idx_ + 1, dtype=acc_dtype_)) * event_bits_in_batch_
                # Compute expected values
                updated_batch_expected_value_ = tf.cast(cumulative_one_bits_, dtype=tf.float64) / tf.cast(cumulative_bits_, dtype=tf.float64)
                updated_expected_value_ = tf.reduce_mean(updated_batch_expected_value_, axis=0)
                # Compute loss
                updated_batch_loss_ = monte_carlo.mse_loss(output_probs_, updated_expected_value_)
                # Write to TensorArray
                losses_ = losses_.write(batch_idx_, updated_batch_loss_)

            return updated_expected_value_, losses_.stack()

        # sample_sizer = {
        #     "batch_size": 16,
        #     "total_batches": 8,
        #     "sample_size": 128,
        #     "num_events": 5,
        # }
        num_events = 8192
        sampler_dtype = tf.float32
        bitpack_dtype = tf.uint8
        optimizer = BatchSampleSizeOptimizer(
            num_events=num_events,
            max_bytes=int(1.8 * 2 ** 32),  # 1.8 times 4 GiB
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
        input_probs = tf.constant([
            [1.0 / (x + 2.0) for x in range(num_events)],
        ], dtype=tf.float32)

        num_outputs = 1
        known_output_probs = tf.constant([
            [1.0 / (x + 2.0) for x in range(num_outputs)],
        ], dtype=tf.float32)

        def build_binary_xor_tree(inputs):
            # inputs: [batch_size, num_events, sample_size]
            current_gates = inputs  # Shape [batch_size, num_gates, sample_size]

            #@tf.function(jit_compile=True)
            def cond(current_gates_):
                num_gates = tf.shape(current_gates_)[1]
                return tf.greater(num_gates, 1)

            def body(current_gates):
                num_gates = tf.shape(current_gates)[1]
                # Determine if num_gates is odd
                is_odd = tf.equal(tf.keras.ops.mod(num_gates, 2), 1)
                pair_num = num_gates // 2
                # Indices for even and odd positions
                even_indices = tf.range(0, pair_num * 2, delta=2)
                odd_indices = tf.range(1, pair_num * 2, delta=2)
                # Gather even and odd gates
                even_gates = tf.gather(current_gates, even_indices, axis=1)
                odd_gates = tf.gather(current_gates, odd_indices, axis=1)
                # XOR them
                new_gates = tf.bitwise.bitwise_xor(even_gates, odd_gates)

                # If num_gates is odd, append the last gate
                def append_last_gate():
                    last_gate = tf.gather(current_gates, [num_gates - 1], axis=1)
                    return tf.concat([new_gates, last_gate], axis=1)

                def do_nothing():
                    return new_gates

                # Update current_gates considering odd/even number of gates
                current_gates = tf.cond(is_odd, append_last_gate, do_nothing)
                return [current_gates]  # Return as list to match loop_vars

            final_gates = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[current_gates],
                shape_invariants=[tf.TensorShape([None, None, None])]
            )

            final_output = tf.squeeze(final_gates[0], axis=1)  # Remove the gates dimension
            return final_output  # Shape [batch_size, sample_size]


        def run():
            est_mean, losses = batched_estimate_outputs(
                input_probs_=tf.broadcast_to(input_probs, [sample_sizer['batch_size'], num_events]),
                output_probs_=tf.broadcast_to(known_output_probs, [sample_sizer['batch_size'], num_outputs]),
                fn_logic=build_binary_xor_tree,
                num_batches_=sample_sizer['total_batches'],
                sample_size_=sample_sizer['sample_size'],
                bitpack_dtype_=tf.uint8,
                sampler_dtype_=tf.float32)

            #print(f"Batch Losses: {losses}")
            #print(f"Known Probabilities ({len(known_output_probs.numpy()[0])}): {known_output_probs.numpy()}")
            #print(f"Estimated Means    ({len(est_mean.numpy())}): {est_mean.numpy()}")
            print(f"Estimated Means: {est_mean.numpy()}")

        count = 10
        times = timeit.timeit(lambda: run(), number=count)
        print(f"total_time (s): {times}")
        print(f"avg_time (s): {times / float(count)}")



if __name__ == "__main__":
    #SubGraphConstructionTests().test_parametrized_seed_and_batch()
    SubGraphConstructionTests().test_generate_bernoulli_and_eval_gates()
    #SubGraphConstructionTests().test_generate_bernoulli()


def build_binary_xor_tree_debug(inputs):
    @tf.function(jit_compile=True)
    def xor(a, b):
        return tf.bitwise.bitwise_xor(a, b)

    # input_shape = [batch_size, num_events, sample_size]
    num_input_events = inputs.shape[1]
    print(f"build_binary_xor_tree inputs.shape: {inputs.shape}")
    level_1_gates = []
    for input_idx in tf.range(num_input_events, delta=2):
        # gate_shape = [batch_size, sample_size]
        l1_gate = xor(inputs[:, input_idx, :], inputs[:, input_idx + 1, :])
        level_1_gates.append(l1_gate)
        print(f"level_1_gates input_idx: {input_idx}, {input_idx + 1}")
    print(f"level_1_gates: {len(level_1_gates)}")

    level_2_gates = []
    for input_idx in tf.range(len(level_1_gates), delta=2):
        print(input_idx, input_idx + 1)
        l2_gate = xor(level_1_gates[input_idx], level_1_gates[input_idx + 1])
        level_2_gates.append(l2_gate)
        print(f"level_2_gates input_idx: {input_idx}, {input_idx + 1}")
    print(f"level_2_gates: {len(level_2_gates)}")

    level_3_gate = xor(level_2_gates[0], level_2_gates[1])
    print(f"level_3_gate : {level_3_gate.shape}, {level_3_gate}")
    return level_3_gate