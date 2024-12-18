import timeit
import unittest

from keras.src.layers import InputLayer
from tensorflow.keras.layers import Concatenate

import numpy as np
import tensorflow as tf

from pracciolini.grammar.canopy.model.layers import BitpackedBernoulli, BitwiseOr, BitwiseXor, Expectation, BitwiseAnd, \
    BitwiseNot, BitwiseXnor, BitwiseNand, BitwiseNor
from pracciolini.grammar.canopy.model.ops.monte_carlo import tally, count_bits
from pracciolini.grammar.canopy.model.ops.sampler import generate_bernoulli, generate_bernoulli_no_bitpack
from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph

import tensorflow as tf
import numpy as np

from pracciolini.grammar.canopy.probability import monte_carlo
from pracciolini.grammar.canopy.utils import compute_optimal_sample_shape_for_constraints, \
    tensor_as_formatted_bit_vectors


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

    def test_parametrized_seed_and_batch(self):
        #tf.profiler.experimental.start('../../../../logs')
        # tf.debugging.experimental.enable_dump_debug_info('../../../../logs', tensor_debug_mode="FULL_HEALTH",
        #                                                  circular_buffer_size=-1)

        tf.config.run_functions_eagerly(False)

        sampler_dtype = tf.float32
        bitpack_dtype = tf.uint8
        num_events = 7
        batch_size, sample_size, extras = compute_optimal_sample_shape_for_constraints(num_events=num_events,
                                                                               max_bytes=int(2**28),
                                                                               dtype=sampler_dtype,
                                                                               batch_size_range=(256, None),
                                                                               sample_size_range=(1,None),
                                                                               bitpack_dtype=bitpack_dtype)
        total_samples = batch_size * sample_size
        total_samples_bit_packed = total_samples * tf.dtypes.as_dtype(bitpack_dtype).size * 8

        datastream_float_type = tf.float32
        datastream_batch_size = batch_size                                # size of each batch (num_batches should be num_samples/batch_size)
        datastream_num_batches = total_samples // datastream_batch_size    # Number of batches

        print(f"N = ({total_samples})[batch_size ({batch_size}) x datastream_num_batches ({datastream_num_batches})]")

        probs = tf.constant([1.0 / (x + 1.0) for x in range(num_events)], dtype=datastream_float_type)

        # Create the dataset
        dataset = tf.data.Dataset.from_tensors(probs).repeat(total_samples)
        batched_dataset = dataset.batch(datastream_batch_size, num_parallel_calls=32)
        batched_dataset = batched_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Define Keras inputs
        probs_input = tf.keras.Input(shape=(num_events,), dtype=datastream_float_type)

        # Create the layers
        samples_layer = BitpackedBernoulli(sample_size=sample_size,
                                           bitpack_dtype=bitpack_dtype,
                                           dtype=sampler_dtype)(probs_input)

        # Build and compile the Keras model
        model = tf.keras.Model(inputs=probs_input, outputs=samples_layer)
        model.compile(run_eagerly=False,
                      steps_per_execution=1,
                      jit_compile=True,)
        model.summary()

        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir="../../../../logs", profile_batch='10, 15')
        # Run the model prediction on the current batch
        model.predict(
            x=batched_dataset,
            steps=datastream_num_batches, # Ensure you cover the entire dataset,
            # callbacks=[tb_callback]
        )
        #tf.profiler.experimental.stop()

        # Iterate over the dataset
      #  for batch_probs in batched_dataset:


        return
        # Define Keras inputs
        probs_input = tf.keras.Input(shape=(width,), dtype=float_type)
        n_sample_packs_input = tf.keras.Input(shape=(), dtype=tf.int32)

        # Create the layers
        samples_layer = BitpackedBernoulli()([probs_input, n_sample_packs_input])
        #expected_values_layer = Expectation()(samples_layer)
        # Build and compile the Keras model
        model = tf.keras.Model(inputs=[probs_input, n_sample_packs_input], outputs=samples_layer)
        model.compile()
        model.summary()

        # Profile from batches 10 to 15
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir="../../../../logs")
        # Run the model prediction
        model.predict(
            x=[probs_array, n_sample_packs_array],
            callbacks=[tb_callback],
            batch_size=batch_size  # Controls how many samples are processed at once
        )
        #print(f"Outputs:\n{outputs}\n")


    def test_generate_bernoulli(self):

        def batched_estimate(probs_: tf.Tensor,
                             num_batches_: int,
                             sample_size_: int,
                             bitpack_dtype_: tf.DType,
                             sampler_dtype_: tf.DType,
                             acc_dtype_: tf.DType = tf.uint32):
            batch_size_ = probs_.shape[0]
            event_dim_size_ = sample_size_ * tf.dtypes.as_dtype(bitpack_dtype_).size * 8
            event_bits_in_batch_ = tf.cast(batch_size_ * event_dim_size_, dtype=acc_dtype_)

            cumulative_one_bits_ = tf.constant(0, dtype=acc_dtype_)
            updated_expected_value_ = tf.constant(0, dtype=tf.float64)
            losses_ = []

            for batch_idx_ in range(num_batches_):
                # Generate samples
                packed_bits_ = generate_bernoulli(
                    probs=probs_,
                    n_sample_packs_per_probability=int(sample_size_),
                    bitpack_dtype=bitpack_dtype_,
                    dtype=sampler_dtype_,
                )
                one_bits_in_batch_ = tf.reduce_sum(input_tensor=tf.cast(x=tf.raw_ops.PopulationCount(x=packed_bits_), dtype=acc_dtype_), axis=-1)
                cumulative_one_bits_ = cumulative_one_bits_ + one_bits_in_batch_
                cumulative_bits_ = (tf.cast(batch_idx_, dtype=acc_dtype_) + 1) * event_bits_in_batch_
                updated_expected_value_ = cumulative_one_bits_ / cumulative_bits_
                updated_loss_ = tf.keras.losses.MSE(probs_, updated_expected_value_,)

                # Compute MSE losses
                losses_.append(updated_loss_.numpy())
                print(f"Batch [{batch_idx_ + 1}] MSE loss: {updated_loss_.numpy()}, E[f(x)]: {updated_expected_value_.numpy()}")

            return losses_, updated_expected_value_

        #tf.profiler.experimental.start('../../../../logs')
        #tf.compat.v1.disable_eager_execution()
        tf.config.run_functions_eagerly(False)

        #sampler tensor_memory = 1, 1024, 8388608
        sampler_dtype = tf.float32
        bitpack_dtype = tf.uint8
        num_events = 8
        probs = tf.constant([
            [1.0 / (x + 2.0) for x in range(num_events)],
        ], dtype=sampler_dtype)

        batch_size, sample_size, extras = compute_optimal_sample_shape_for_constraints(num_events=num_events,
                                                                               max_bytes=int(1.5 * 2**32),
                                                                               dtype=sampler_dtype,
                                                                               batch_size_range=(2, None),
                                                                               sample_size_range=(2, None),
                                                                               bitpack_dtype=bitpack_dtype)
        print(f"sample_size: {sample_size}, batch_size: {batch_size}")
        print(extras)

        # Initialize cumulative statistics
        total_batches = 5
        losses, est_mean = batched_estimate(probs_=probs,
                         num_batches_=total_batches,
                         sample_size_=sample_size,
                         bitpack_dtype_=bitpack_dtype,
                         sampler_dtype_=sampler_dtype)


        print(f"Losses: {losses}")
        print(f"Known Probabilities: {probs.numpy()}")
        print(f"Estimated Means    : {est_mean.numpy()}")
        #
        # import matplotlib.pyplot as plt
        # # Plot the decreasing loss trends
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, num_batches + 1), cumulative_losses, marker='o')
        # plt.title('MSE Loss vs. Number of Batches Processed')
        # plt.xlabel('Number of Batches')
        # plt.ylabel('MSE Loss')
        # plt.grid(True)
        # plt.show()
        return
        packed_bits = generate_bernoulli(
            probs=probs,
            n_sample_packs_per_probability=int(sample_size),
            bitpack_dtype=bitpack_dtype,
            dtype=sampler_dtype,
        )

        samples_per_batch = sample_size * tf.dtypes.as_dtype(bitpack_dtype).size * 8
        total_samples = batch_size * samples_per_batch
        print(f"N = ({total_samples})[num_batches ({batch_size}) x samples_per_batch ({samples_per_batch})]")
        lower_limit, expected_value, upper_limit = tally(packed_bits)
        losses = tf.keras.losses.MSE(probs, expected_value)
        print(f"MSE losses: {losses}")

if __name__ == "__main__":
    #SubGraphConstructionTests().test_parametrized_seed_and_batch()
    SubGraphConstructionTests().test_generate_bernoulli()


