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

@tf.function
def sample_bernoulli():
    pass

@tf.function
def sample_bitpack_bernoulli():
    pass

class BitpackSamplesMixin(object):

    def __init__(self, pack_bits_dtype:tf.DType = tf.uint8, *args, **kwargs):

        self.bitpack_dtype = pack_bits_dtype
        self.bitpack_largest_supported_dtype = tf.uint64
        self.bitpack_bits_per_dtype = tf.dtypes.as_dtype(self.bitpack_dtype).size * 8
        self.bitpack_bits_per_largest_supported_dtype = tf.dtypes.as_dtype(self.bitpack_largest_supported_dtype).size * 8

        if self.bitpack_bits_per_dtype > self.bitpack_bits_per_largest_supported_dtype:
            raise ValueError(
                f"Cannot handle word-size {self.bitpack_dtype}, which is larger than {self.bitpack_largest_supported_dtype}"
            )

        # Precompute weights
        weights = [1 << (self.bitpack_bits_per_dtype - 1 - i) for i in range(self.bitpack_bits_per_dtype)]
        compute_dtype = self.bitpack_largest_supported_dtype
        self._weights = tf.constant(weights, dtype=compute_dtype)


    def _define_precompute_weights(self):
        weights = [1<<(self.bitpack_bits_per_dtype-1-i) for i in range(self.bitpack_bits_per_dtype)]
        compute_dtype = self.bitpack_largest_supported_dtype
        @tf.function
        def precompute_weights():
            return tf.constant(weights, dtype=compute_dtype)
        return precompute_weights

    def sample(self, sample_shape=(), seed=None, **kwargs):
        """Generates samples, packing bits along the sample dimensions.

        Args:
            sample_shape: Shape of the samples to draw.
            seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
            **kwargs: Named arguments forwarded to subclass implementation.

        Returns:
            Samples from the distribution, with bits packed into integers.
        """

        # Obtain samples from the base class (tfp.distributions.Distribution)
        samples = super().sample(sample_shape=sample_shape, seed=seed, **kwargs)

        # Convert samples to booleans (since we initialized with dtype=tf.uint8)
        bool_tensor = tf.bitcast(samples, tf.uint8)

        # Flatten sample and event dimensions for bit-packing
        samples_shape = tf.shape(bool_tensor)
        samples_rank = tf.rank(bool_tensor)

        batch_shape_tensor = tf.shape(self.batch_shape_tensor())
        batch_rank = tf.size(self.batch_shape_tensor())

        event_shape_tensor = tf.shape(self.event_shape_tensor())
        event_rank = tf.size(self.event_shape_tensor())

        sample_ndims = samples_rank - batch_rank - event_rank

        # Reshape to [batch_size, total_bits]
        sample_shape_tensor = samples_shape[:sample_ndims]
        batch_shape_tensor = samples_shape[sample_ndims: sample_ndims + batch_rank]
        event_shape_tensor = samples_shape[sample_ndims + batch_rank:]

        sample_size = tf.reduce_prod(sample_shape_tensor)
        batch_size = tf.reduce_prod(batch_shape_tensor)
        event_size = tf.reduce_prod(event_shape_tensor)

        total_bits = sample_size * event_size

        bool_tensor_flat = tf.reshape(bool_tensor, [batch_size, total_bits])

        # Pack the bits
        packed_ints = BitpackSamplesMixin._pack_tensor_bits_impl(bool_tensor_flat, self._weights)

        # Reshape to batch dimensions
        num_packed_elements = tf.shape(packed_ints)[1]

        if batch_rank > 0:
            final_shape = tf.concat([batch_shape_tensor, [num_packed_elements]], axis=0)
        else:
            final_shape = [num_packed_elements]

        packed_tensor = tf.reshape(packed_ints, final_shape)
        packed_tensor = tf.cast(packed_tensor, dtype=self.bitpack_dtype)

        return packed_tensor

    @staticmethod
    @tf.function
    def _pack_tensor_bits_impl(bool_tensor: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """
        Packs bits using TensorFlow operations, with padding if necessary.

        Args:
            bool_tensor: A boolean tensor of shape [batch_size, total_bits].
            weights: A tensor containing weights for each bit position.

        Returns:
            A tensor containing packed integers of shape [batch_size, num_packed_elements].
        """
        compute_dtype = weights.dtype

        n_bits = tf.shape(weights)[0]
        total_bits = tf.shape(bool_tensor)[-1]

        # Calculate padding if total_bits is not a multiple of n_bits
        remaining_bits = total_bits % n_bits
        if remaining_bits > 0:
            pad_size = n_bits - remaining_bits
            bool_tensor = tf.pad(bool_tensor, [[0, 0], [0, pad_size]], constant_values=False)

        # Reshape and pack
        reshaped_tensor = tf.reshape(bool_tensor, [tf.shape(bool_tensor)[0], -1, n_bits])
        bits = tf.cast(reshaped_tensor, dtype=compute_dtype)
        weighted_bits = bits * weights

        # Sum over the bits to get packed integers
        packed_ints = tf.reduce_sum(weighted_bits, axis=-1)

        return packed_ints

class Bernoulli(BitpackSamplesMixin, tfp.distributions.Bernoulli):
    def __init__(self, *args, pack_bits_dtype=tf.uint8, **kwargs):
        # Initialize the Bernoulli distribution with dtype=tf.uint8
        tfp.distributions.Bernoulli.__init__(self, *args, dtype=tf.uint8, **kwargs)
        # Initialize BitpackSamplesMixin
        BitpackSamplesMixin.__init__(self, pack_bits_dtype=pack_bits_dtype)

class Binomial(BitpackSamplesMixin, tfp.distributions.Binomial):
    def __init__(self, *args, pack_bits_dtype=tf.uint8, **kwargs):
        # Initialize the Binomial distribution with dtype=tf.uint8
        tfp.distributions.Binomial.__init__(self, *args, dtype=tf.uint8, **kwargs)
        # Initialize BitpackSamplesMixin
        BitpackSamplesMixin.__init__(self, pack_bits_dtype=pack_bits_dtype)

class Categorical(BitpackSamplesMixin, tfp.distributions.Categorical):
    def __init__(self, *args, pack_bits_dtype=tf.uint8, **kwargs):
        # Initialize the Categorical distribution with dtype=tf.uint8
        tfp.distributions.Categorical.__init__(self, *args, dtype=tf.uint8, **kwargs)
        # Initialize BitpackSamplesMixin
        BitpackSamplesMixin.__init__(self, pack_bits_dtype=pack_bits_dtype)