import timeit

import tensorflow as tf

from pracciolini.grammar.canopy.module.sampler import Sampler
from pracciolini.grammar.canopy.module.tree_builder import build_binary_xor_tree


class LogicTreeBroadcastSampler(Sampler):
    def __init__(self, logic_fn, num_inputs, num_outputs, num_batches, batch_size, sample_size,
                 bitpack_dtype: tf.uint8,
                 sampler_dtype: tf.float32,
                 acc_dtype: tf.float32):
        tf.config.run_functions_eagerly(False)
        super().__init__()

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._sample_size = sample_size
        self._bitpack_dtype = bitpack_dtype
        self._sampler_dtype = sampler_dtype
        self._acc_dtype = acc_dtype

        self._num_sampled_bits_in_batch = tf.constant(value=tf.cast(self._batch_size * self._sample_size * super(LogicTreeBroadcastSampler, self)._compute_bits_in_dtype(self._bitpack_dtype), dtype=acc_dtype), dtype=acc_dtype)

        self._mse_loss = tf.function(
            func=lambda y_true, y_pred: super(LogicTreeBroadcastSampler, self)._mse_loss(y_true, y_pred, dtype=self._acc_dtype),
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._acc_dtype, name='y_true'),
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._acc_dtype, name='y_pred'),
            ],
            jit_compile=True
        )

        self._generate_bernoulli_batch = tf.function(
            func=lambda probs, seed: super(LogicTreeBroadcastSampler, self)._generate_bernoulli(
                probs=probs,
                n_sample_packs_per_probability=tf.constant(value=self._sample_size, dtype=tf.int32) ,
                bitpack_dtype=self._bitpack_dtype,
                dtype=self._sampler_dtype,
                seed=seed
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs, self._batch_size], dtype=self._sampler_dtype, name='probs'),
                tf.TensorSpec(shape=[], dtype=tf.int32, name='seed'),
            ],
            jit_compile=True
        )

        self._generate_bernoulli_broadcast_no_batch = tf.function(
            func=lambda probs, seed: super(LogicTreeBroadcastSampler, self)._generate_bernoulli(
                probs=tf.broadcast_to(tf.expand_dims(probs, axis=1), [self._num_inputs, self._batch_size]),
                n_sample_packs_per_probability=tf.constant(value=self._sample_size, dtype=tf.int32) ,
                bitpack_dtype=self._bitpack_dtype,
                dtype=self._sampler_dtype,
                seed=seed
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs], dtype=self._sampler_dtype, name='probs'),
                tf.TensorSpec(shape=[], dtype=tf.int32, name='seed'),
            ],
            jit_compile=True
        )

        self._logic_fn = tf.function(
            func=logic_fn,
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs, self._batch_size, self._sample_size], dtype=self._bitpack_dtype, name='sampled_inputs'),
            ],
            jit_compile=True
        )

        self._count = tf.function(
            func=lambda x: super(LogicTreeBroadcastSampler, self)._count_one_bits(
                x=x,
                axis=None,
                dtype=tf.uint32,
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._bitpack_dtype, name='raw_bits'),
            ],
            jit_compile=True
        )

        self._tally = tf.function(
            func=lambda means: super(LogicTreeBroadcastSampler, self)._p95_ci(
                means=means,
                total=self._num_sampled_bits_in_batch,
                dtype=self._acc_dtype,
            ),
            input_signature=[
                tf.TensorSpec(shape=[], dtype=self._acc_dtype, name='means'),
            ],
            jit_compile=True
        )

    @tf.function(jit_compile=True)
    def generate(self, probs, seed=372):
        return self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)

    @tf.function(jit_compile=True)
    def eval(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        return output_packed_bits_

    @tf.function(jit_compile=True)
    def count(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        ones_ = self._count(output_packed_bits_)
        return ones_

    @tf.function(jit_compile=True)
    def expectation(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        ones_ = self._count(output_packed_bits_)
        means_ = tf.cast(ones_, dtype=self._acc_dtype) / self._num_sampled_bits_in_batch
        return means_

    @tf.function(jit_compile=True)
    def tally(self, probs, seed=372):
        input_packed_bits_ = self._generate_bernoulli_broadcast_no_batch(probs=probs, seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        ones_ = self._count(output_packed_bits_)
        means_ = tf.cast(ones_, dtype=self._acc_dtype) / self._num_sampled_bits_in_batch
        p05_, p95_ = self._tally(means_)
        return p05_, means_, p95_




if __name__ == "__main__":
    @tf.function(jit_compile=True)
    def some_logic_expression(inputs):
        # input_shape = [batch_size, num_events, sample_size]
        # print(inputs.shape)
        g1 = tf.bitwise.bitwise_or(inputs[0, :, :], inputs[3, :, :])
        g2 = tf.bitwise.bitwise_xor(inputs[4, :, :], g1)
        g3 = tf.bitwise.bitwise_and(g1, g2)
        outputs = g3
        # output_shape = [batch_size, sample_size] (for a single output)
        # to be
        return outputs


    num_events = 1024

    input_probs = (tf.constant([1.0 / (x + 2.0) for x in range(num_events)], dtype=tf.float32))

    sampler = LogicTreeBroadcastSampler(
        logic_fn=build_binary_xor_tree,
        num_inputs=num_events,
        num_outputs=1,
        num_batches=2,
        batch_size=474,
        sample_size=474,
        bitpack_dtype=tf.uint8,
        sampler_dtype=tf.float32,
        acc_dtype=tf.float32
    )

    count = 100
    t = timeit.Timer(lambda: sampler.tally(input_probs))
    times = t.timeit(number=count)
    print(f"total_time (s): {times}")
    print(f"avg_time (s): {times / float(count)}")
    print(sampler.tally(input_probs))
    print(sampler.tally(input_probs))