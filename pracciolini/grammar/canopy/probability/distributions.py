from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
"""
  #### Shapes

  There are three important concepts associated with TensorFlow Distributions
  shapes:

  - Event shape describes the shape of a single draw from the distribution;
    it may be dependent across dimensions. For scalar distributions, the event
    shape is `[]`. For a 5-dimensional MultivariateNormal, the event shape is
    `[5]`.
  - Batch shape describes independent, not identically distributed draws, aka a
    "collection" or "bunch" of distributions.
  - Sample shape describes independent, identically distributed draws of batches
    from the distribution family.

  The event shape and the batch shape are properties of a Distribution object,
  whereas the sample shape is associated with a specific call to `sample` or
  `log_prob`.

  For detailed usage examples of TensorFlow Distributions shapes, see
  [this tutorial](
  https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb)

  #### Broadcasting, batching, and shapes

  All distributions support batches of independent distributions of that type.
  The batch shape is determined by broadcasting together the parameters.

  The shape of arguments to `__init__`, `cdf`, `log_cdf`, `prob`, and
  `log_prob` reflect this broadcasting, as does the return value of `sample`.

  `sample_n_shape = [n] + batch_shape + event_shape`, where `sample_n_shape` is
  the shape of the `Tensor` returned from `sample(n)`, `n` is the number of
  samples, `batch_shape` defines how many independent distributions there are,
  and `event_shape` defines the shape of samples from each of those independent
  distributions. Samples are independent along the `batch_shape` dimensions, but
  not necessarily so along the `event_shape` dimensions (depending on the
  particulars of the underlying distribution).

  Using the `Uniform` distribution as an example:

  ```python
  minval = 3.0
  maxval = [[4.0, 6.0],
            [10.0, 12.0]]

  # Broadcasting:
  # This instance represents 4 Uniform distributions. Each has a lower bound at
  # 3.0 as the `minval` parameter was broadcasted to match `maxval`'s shape.
  u = Uniform(minval, maxval)

  # `event_shape` is `TensorShape([])`.
  event_shape = u.event_shape
  # `event_shape_t` is a `Tensor` which will evaluate to [].
  event_shape_t = u.event_shape_tensor()

  # Sampling returns a sample per distribution. `samples` has shape
  # [5, 2, 2], which is [n] + batch_shape + event_shape, where n=5,
  # batch_shape=[2, 2], and event_shape=[].
  samples = u.sample(5)

  # The broadcasting holds across methods. Here we use `cdf` as an example. The
  # same holds for `log_cdf` and the likelihood functions.

  # `cum_prob` has shape [2, 2] as the `value` argument was broadcasted to the
  # shape of the `Uniform` instance.
  cum_prob_broadcast = u.cdf(4.0)

  # `cum_prob`'s shape is [2, 2], one per distribution. No broadcasting
  # occurred.
  cum_prob_per_dist = u.cdf([[4.0, 5.0],
                             [6.0, 7.0]])

  # INVALID as the `value` argument is not broadcastable to the distribution's
  # shape.
  cum_prob_invalid = u.cdf([4.0, 5.0, 6.0])
  ```

"""

class BitpackSamplesMixin(object):

    def __init__(self, dtype:tf.DType = tf.uint64, *args, **kwargs):
        # always initialize the actual distribution as 8-bit unsigned int.
        super().__init__(dtype=tf.uint8, *args, **kwargs)

        self._bitpack_dtype = dtype
        self._bitpack_largest_supported_dtype = tf.uint64
        self._bitpack_bits_per_dtype = tf.dtypes.as_dtype(self._bitpack_dtype).size * 8
        self._bitpack_bits_per_largest_supported_dtype = tf.dtypes.as_dtype(self._bitpack_largest_supported_dtype).size * 8

        if self._bitpack_bits_per_dtype > self._bitpack_bits_per_largest_supported_dtype:
            raise ValueError(f"Cannot handle word-size {self._bitpack_dtype}, which is larger than {self._bitpack_largest_supported_dtype}")

        self.precompute_weights = self._define_precompute_weights()()


    def _define_precompute_weights(self):
        weights = [1<<(self._bitpack_bits_per_dtype-1-i) for i in range(self._bitpack_bits_per_dtype)]
        compute_dtype = self._bitpack_largest_supported_dtype
        @tf.function
        def precompute_weights():
            return tf.constant(weights, dtype=compute_dtype)
        return precompute_weights


    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        """Generates samples of the specified shape, optionally packing bits along sample dimensions.

        Args:
            sample_shape: Shape of the samples to draw.
            seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
            name: name to give to the op.
            **kwargs: Named arguments forwarded to subclass implementation.

        Returns:
            Samples from the distribution, potentially with bits packed into integers.
        """
        _sample_shape = sample_shape
        _n_bits = self._bitpack_bits_per_dtype
        if sample_shape == () or sample_shape == []:
            _sample_shape = [_n_bits]
        elif isinstance(sample_shape, int):
            _sample_shape = [sample_shape]
        elif isinstance(sample_shape, tuple):
            _sample_shape = list(sample_shape)

        padding = _sample_shape[0] % _n_bits
        if padding != 0:
            _sample_shape[0] = _n_bits if _n_bits > _sample_shape[0] else _sample_shape[0] - padding

        samples = super().sample(sample_shape=_sample_shape, seed=seed, name=name, **kwargs)

        with tf.name_scope(name or 'sample'):

            # Convert samples to booleans
            #bool_tensor = tf.cast(samples, tf.bool)
            #bool_tensor = samples

            # We need to reshape bool_tensor appropriately
            # First, get the shapes
            samples_shape = tf.shape(samples)
            samples_rank = tf.rank(samples)

            # Get the batch shape tensor
            batch_shape_tensor = self.batch_shape_tensor()
            batch_shape = tf.shape(batch_shape_tensor)
            batch_rank = tf.size(batch_shape_tensor)

            # Sample dimensions are the leading dimensions in samples
            sample_ndims = samples_rank - batch_rank

            # If sample_ndims == 0, we cannot pack bits
            with tf.control_dependencies([
                tf.debugging.assert_greater_equal(sample_ndims, 1,
                                                  message="Sample dimensions must be at least 1 when packing bits")
            ]):
                sample_shape_tensor = samples_shape[:sample_ndims]
                batch_shape_tensor = samples_shape[sample_ndims:]

                total_bits = tf.reduce_prod(sample_shape_tensor)
                total_batch_size = tf.reduce_prod(batch_shape_tensor)

                # Reshape bool_tensor to [total_batch_size, total_bits]
                new_shape = tf.concat([[total_batch_size], [total_bits]], axis=0)
                bool_tensor_reshaped = tf.reshape(samples, new_shape)

                # Pack bits using _pack_tensor_bits_impl
                packed_ints = BitpackSamplesMixin._pack_tensor_bits_no_padding_impl(bool_tensor_reshaped, self.precompute_weights)

                # Now, we need to reshape the output to appropriate shape
                num_packed_elements_per_row = ((total_bits + _n_bits - 1) // _n_bits)

                # Output shape: batch_shape + [num_packed_elements_per_row]
                output_shape = tf.concat([batch_shape_tensor, [num_packed_elements_per_row]], axis=0)
                packed_tensor = tf.reshape(packed_ints, output_shape)

                # Cast to desired dtype
                packed_tensor = tf.cast(packed_tensor, dtype=self._bitpack_dtype)

                return packed_tensor

    @staticmethod
    @tf.function
    def _pack_tensor_bits_no_padding_impl(bool_tensor: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """
        Internal function to pack bits using TensorFlow operations.

        Args:
            bool_tensor: A boolean tensor.
            weights: A tensor containing weights for each bit position.

        Returns:
            A tensor containing packed integers.
        """

        # Convert bits to integers and apply weights
        bits = tf.cast(bool_tensor, dtype=weights.dtype)
        weighted_bits = bits * weights

        # Sum over the bits to get packed integers
        packed_ints = tf.reduce_sum(weighted_bits, axis=-1)

        return packed_ints

    @staticmethod
    @tf.function
    def _pack_tensor_bits_impl(bool_tensor: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """
        Internal function to pack bits using TensorFlow operations.

        Args:
            bool_tensor: A boolean tensor.
            weights: A tensor containing weights for each bit position.

        Returns:
            A tensor containing packed integers.
        """
        compute_dtype = tf.uint64

        n_bits = tf.shape(weights)[0]

        # Compute the total number of bits after padding
        total_bits = tf.shape(bool_tensor)[-1]
        y_padded = ((total_bits + n_bits - 1) // n_bits) * n_bits  # Round up to nearest multiple of n_bits
        pad_size = y_padded - total_bits  # Number of bits to pad

        # Pad the boolean tensor on the right with False (equivalent to 0)
        padded_tensor = tf.pad(
            bool_tensor, [[0, 0], [0, pad_size]], constant_values=False
        )

        # Reshape to (batch_size, num_packed_elements_per_row, n_bits)
        reshaped_tensor = tf.reshape(padded_tensor, [tf.shape(padded_tensor)[0], -1, n_bits])

        # Convert bits to integers and apply weights
        bits = tf.cast(reshaped_tensor, dtype=compute_dtype)
        weighted_bits = bits * weights

        # Sum over the bits to get packed integers
        packed_ints = tf.reduce_sum(weighted_bits, axis=-1)

        return packed_ints


class Bernoulli(BitpackSamplesMixin, tfp.distributions.Bernoulli):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Binomial(BitpackSamplesMixin, tfp.distributions.Binomial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Categorical(BitpackSamplesMixin, tfp.distributions.Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
