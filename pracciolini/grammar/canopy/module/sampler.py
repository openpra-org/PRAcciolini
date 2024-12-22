import tensorflow as tf

from pracciolini.grammar.canopy.model.ops.sampler import generate_bernoulli
from pracciolini.grammar.canopy.probability.monte_carlo import mse_loss


class LogicTreeSampler(tf.Module):
    def __init__(self, logic_fn, num_inputs, num_outputs, num_batches, batch_size, sample_size,
                 bitpack_dtype: tf.uint8,
                 sampler_dtype: tf.float32,
                 acc_dtype: tf.float32):
        super().__init__()
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._sample_size = sample_size
        self._bitpack_dtype = bitpack_dtype
        self._sampler_dtype = sampler_dtype
        self._acc_dtype = acc_dtype

        self._mse_loss = tf.function(
            func=lambda y_true, y_pred: mse_loss(y_true, y_pred, dtype=self._acc_dtype),
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._acc_dtype, name='y_true'),
                tf.TensorSpec(shape=[self._batch_size, self._sample_size], dtype=self._acc_dtype, name='y_pred'),
            ],
            jit_compile=True
        )

        self.generate_bernoulli_batch = tf.function(
            func=lambda probs, seed: generate_bernoulli(
                probs=probs,
                n_sample_packs_per_probability=tf.constant(value=self._sample_size, dtype=tf.int32) ,
                bitpack_dtype=self._bitpack_dtype,
                dtype=self._sampler_dtype,
                seed=seed
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._num_inputs], dtype=self._sampler_dtype, name='input_probs'),
                tf.TensorSpec(shape=[], dtype=tf.int32, name='seed'),
            ],
            jit_compile=True
        )

        self.generate_bernoulli_broadcast_no_batch = tf.function(
            func=lambda probs, seed: generate_bernoulli(
                probs=tf.broadcast_to(probs, [self._batch_size, self._num_inputs]),
                n_sample_packs_per_probability=tf.constant(value=self._sample_size, dtype=tf.int32) ,
                bitpack_dtype=self._bitpack_dtype,
                dtype=self._sampler_dtype,
                seed=seed
            ),
            input_signature=[
                tf.TensorSpec(shape=[self._num_inputs], dtype=self._sampler_dtype, name='input_probs'),
                tf.TensorSpec(shape=[], dtype=tf.int32, name='seed'),
            ],
            jit_compile=True
        )

        self._logic_fn = tf.function(
            func=logic_fn,
            input_signature=[
                tf.TensorSpec(shape=[self._batch_size, self._num_inputs, self._sample_size], dtype=self._bitpack_dtype, name='sampled_inputs'),
            ],
            jit_compile=True
        )

    @tf.function(jit_compile=True)
    def generate(self, input_probs_, seed=372):
        return self.generate_bernoulli_broadcast_no_batch(probs=input_probs_, seed=seed,)

    @tf.function(jit_compile=True)
    def generate_batch(self, batch_input_probs_, seed=372):
        return self.generate_bernoulli_batch(probs=batch_input_probs_, seed=seed,)

    @tf.function(jit_compile=True)
    def sample(self, input_probs_, seed=372):
        input_packed_bits_ = self.generate_bernoulli_broadcast_no_batch(probs=input_probs_,seed=seed,)
        output_packed_bits_ = self._logic_fn(input_packed_bits_)
        return output_packed_bits_



if __name__ == "__main__":
    def some_logic_expression(inputs):
        # input_shape = [batch_size, num_events, sample_size]
        # print(inputs.shape)
        g1 = tf.bitwise.bitwise_or(inputs[:, 0, :], inputs[:, 3, :])
        g2 = tf.bitwise.bitwise_xor(inputs[:, 4, :], g1)
        g3 = tf.bitwise.bitwise_and(g1, g2)
        outputs = g3
        # output_shape = [batch_size, sample_size] (for a single output)
        # to be
        return outputs


    num_events = 5

    input_probs = tf.constant([1.0 / (x + 2.0) for x in range(num_events)], dtype=tf.float32)

    sampler = LogicTreeSampler(
        logic_fn=some_logic_expression,
        num_inputs=num_events,
        num_outputs=1,
        num_batches=2,
        batch_size=1024,
        sample_size=1024,
        bitpack_dtype=tf.uint8,
        sampler_dtype=tf.float32,
        acc_dtype=tf.float32
    )

    output_bits = sampler.sample(input_probs)
    print(output_bits)