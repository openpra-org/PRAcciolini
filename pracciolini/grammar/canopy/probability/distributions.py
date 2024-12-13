from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

class BitpackSamplesMixin(object):
    def sample(self, sample_shape=(), seed=None, name='sample', pack_bits: Optional[tf.DType] = None, **kwargs):
        """Generates samples of the specified shape, optionally packing bits along sample dimensions.

        Args:
            sample_shape: Shape of the samples to draw.
            seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
            name: name to give to the op.
            pack_bits: Optional tf.DType to pack bits into. If specified, samples are packed into integers of this type.
            **kwargs: Named arguments forwarded to subclass implementation.

        Returns:
            Samples from the distribution, potentially with bits packed into integers.
        """
        samples = super().sample(sample_shape=sample_shape, seed=seed, name=name, **kwargs)
        if pack_bits is None:
            return samples

        with tf.name_scope(name or 'sample'):
            # Convert samples to booleans
            bool_tensor = tf.cast(samples, tf.bool)

            # Get the number of bits in the pack_bits dtype
            n_bits = tf.dtypes.as_dtype(pack_bits).size * 8  # Number of bits in target dtype

            if n_bits > tf.dtypes.as_dtype(tf.uint64).size * 8:
                raise ValueError("Cannot handle word-size larger than tf.uint64")

            compute_dtype = tf.uint64

            # Precompute weights
            weights_list = [1 << (n_bits - 1 - i) for i in range(n_bits)]
            weights = tf.constant(weights_list, dtype=compute_dtype)

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
                bool_tensor_reshaped = tf.reshape(bool_tensor, new_shape)

                # Pack bits using _pack_tensor_bits_impl
                packed_ints = BitpackSamplesMixin._pack_tensor_bits_impl(bool_tensor_reshaped, weights)

                # Now, we need to reshape the output to appropriate shape
                num_packed_elements_per_row = ((total_bits + n_bits - 1) // n_bits)

                # Output shape: batch_shape + [num_packed_elements_per_row]
                output_shape = tf.concat([batch_shape_tensor, [num_packed_elements_per_row]], axis=0)
                packed_tensor = tf.reshape(packed_ints, output_shape)

                # Cast to desired dtype
                packed_tensor = tf.cast(packed_tensor, dtype=pack_bits)

                return packed_tensor

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


class Beta(BitpackSamplesMixin, tfp.distributions.Beta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Binomial(BitpackSamplesMixin, tfp.distributions.Binomial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Categorical(BitpackSamplesMixin, tfp.distributions.Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Empirical(BitpackSamplesMixin, tfp.distributions.Empirical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Exponential(BitpackSamplesMixin, tfp.distributions.Exponential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Gamma(BitpackSamplesMixin, tfp.distributions.Gamma):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class JointDistribution(BitpackSamplesMixin, tfp.distributions.JointDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LogNormal(BitpackSamplesMixin, tfp.distributions.LogNormal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Normal(BitpackSamplesMixin, tfp.distributions.Normal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PERT(BitpackSamplesMixin, tfp.distributions.PERT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Poisson(BitpackSamplesMixin, tfp.distributions.Poisson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QuantizedDistribution(BitpackSamplesMixin, tfp.distributions.QuantizedDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Sample(BitpackSamplesMixin, tfp.distributions.Sample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TruncatedNormal(BitpackSamplesMixin, tfp.distributions.TruncatedNormal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Uniform(BitpackSamplesMixin, tfp.distributions.Uniform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Weibull(BitpackSamplesMixin, tfp.distributions.Weibull):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
