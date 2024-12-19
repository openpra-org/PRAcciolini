from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import tensorflow as tf
from sympy.physics.units import momentum


def compute_bits_in_dtype(tensor_type: tf.DType) -> int:
    """
    Computes the number of bits in the given data type.

    Args:
        tensor_type (tf.DType): The tensor data type.

    Returns:
        int: Number of bits in the data type.
    """
    return tf.dtypes.as_dtype(tensor_type).size * 8

import tensorflow as tf

@tf.function
def optimize_batch_and_sample_size(
    num_events: int,
    max_bytes: int,
    sampled_bits_per_event_range: (int, int),
    sampler_dtype: tf.DType = tf.float32,
    bitpack_dtype: tf.DType = tf.uint8,
    batch_size_range: (int, int) = (1, None),
    sample_size_range: (int, int) = (1, None),
    total_batches_range: (int, int) = (1, None),
    learning_rate: float = 1.0,
    max_iterations: int = 500,
    tolerance: float = 1e-4,
):
    """
    Optimizes the batch_size and sample_size to maximize processing efficiency
    under memory constraints using TensorFlow.

    Args:
        num_events (int): Total number of events/probabilities.
        max_bytes (int): Maximum allowed bytes for memory usage per batch.
        sampler_dtype (tf.DType): Bits in data type for event data (e.g., 32 for float32).
        bitpack_dtype (tf.DType): Bits in data type for bit-packing (e.g., 8 for uint8).
        sampled_bits_per_event_range (tuple): Desired cumulative bits per event (min, max).
        batch_size_range (tuple): (batch_size_min, batch_size_max).
        sample_size_range (tuple): (sample_size_min, sample_size_max).
        total_batches_range (tuple): (total_batches_min, total_batches_max).
        learning_rate (float): Learning rate for the optimizer.
        max_iterations (int): Maximum number of iterations for the optimizer.
        tolerance (float): Tolerance for stopping criteria.

    Returns:
        dict: A dictionary containing the optimal values and related computations.
    """
    # Convert inputs to TensorFlow constants

    num_events = tf.constant(num_events, dtype=tf.float32)
    max_bits = tf.constant(max_bytes * 8, dtype=tf.float32)  # Convert max_bytes to bits
    bits_in_sampler_dtype = tf.constant(compute_bits_in_dtype(sampler_dtype), dtype=tf.float32)
    bits_in_bitpack_dtype = tf.constant(compute_bits_in_dtype(bitpack_dtype), dtype=tf.float32)

    # Variable bounds
    sampled_bits_per_event_min = tf.constant(bits_in_bitpack_dtype if sampled_bits_per_event_range[0] is None else sampled_bits_per_event_range[0], dtype=tf.float32)
    sampled_bits_per_event_max = tf.constant(2**31 - 1 if sampled_bits_per_event_range[1] is None else sampled_bits_per_event_range[1], dtype=tf.float32)

    batch_size_min = tf.constant(batch_size_range[0], dtype=tf.float32)
    batch_size_max = tf.constant(batch_size_range[1] if batch_size_range[1] is not None else int(sampled_bits_per_event_max.numpy() / bits_in_bitpack_dtype.numpy()), dtype=tf.float32)

    sample_size_min = tf.constant(sample_size_range[0], dtype=tf.float32)
    sample_size_max = tf.constant(sample_size_range[1] if sample_size_range[1] is not None else int(sampled_bits_per_event_max.numpy() / bits_in_bitpack_dtype.numpy()), dtype=tf.float32)

    total_batches_min = tf.constant(total_batches_range[0], dtype=tf.float32)
    total_batches_max = tf.constant(total_batches_range[1] if total_batches_range[1] is not None else int(sampled_bits_per_event_max.numpy() / bits_in_bitpack_dtype.numpy()),  dtype=tf.float32)

    # Initialize decision variables as TensorFlow variables with initial guesses
    batch_size = tf.Variable(batch_size_min, trainable=True, dtype=tf.float32, name='batch_size')
    sample_size = tf.Variable(sample_size_min, trainable=True, dtype=tf.float32, name='sample_size')
    total_batches = tf.Variable(total_batches_min, trainable=True, dtype=tf.float32, name='total_batches')

    # Use GradientTape for automatic differentiation
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.8)

    @tf.function
    def bounds_penalty(x_min_, x_, x_max_):
        return tf.math.square(tf.nn.relu(x_min_ - x_)) + tf.math.square(tf.nn.relu(x_ - x_max_))

    @tf.function
    def compute_forward_pass(total_batches_, batch_size_, num_events_, sample_size_, bits_in_bitpack_dtype_, bits_in_sampler_dtype_):
        total_sampled_bits_per_event_ = tf.math.reduce_prod([total_batches_, batch_size_, sample_size_, bits_in_bitpack_dtype_])
        bitpack_allocated_bits_in_batch_ = tf.math.reduce_prod([batch_size_, num_events_, sample_size_, bits_in_bitpack_dtype_])
        sampler_allocated_bits_in_batch_ = tf.math.reduce_prod([batch_size_, num_events_, sample_size_, bits_in_sampler_dtype_])
        total_allocated_bits_in_batch_ = tf.math.add(bitpack_allocated_bits_in_batch_, sampler_allocated_bits_in_batch_)
        return total_sampled_bits_per_event_, total_allocated_bits_in_batch_

    prev_loss = 0
    wait_time = 0
    patience = 10
    for iteration in range(max_iterations):
        with tf.GradientTape() as tape:

            total_sampled_bits_per_event, total_allocated_bits_in_batch = compute_forward_pass(total_batches, batch_size, num_events, sample_size, bits_in_bitpack_dtype, bits_in_sampler_dtype)

            memory_size_penalty = bounds_penalty(1.0, total_allocated_bits_in_batch, max_bits)
            batch_size_penalty = bounds_penalty(batch_size_min, batch_size, batch_size_max)
            sample_size_penalty = bounds_penalty(sample_size_min, sample_size, sample_size_max)
            total_batches_penalty = bounds_penalty(total_batches_min, total_batches, total_batches_max)
            total_sampled_bits_penalty = bounds_penalty(sampled_bits_per_event_min, total_sampled_bits_per_event, sampled_bits_per_event_max)
            penalties = tf.math.reduce_sum([memory_size_penalty, batch_size_penalty, sample_size_penalty, total_batches_penalty, total_sampled_bits_penalty])

            # maximize sample_size, batch_size, total_sampled_bits_per_event
            objective = 0.
            #objective = tf.math.add(objective, tf.math.multiply(-1.0, batch_size))
            #objective = tf.math.add(objective, tf.math.multiply(-1.0, sample_size))
            objective = tf.math.add(objective, tf.math.multiply(-1.0, total_sampled_bits_per_event))
            # minimize the total batches
            #objective = tf.math.add(objective, tf.math.multiply(+1e, total_allocated_bits_in_batch))
            objective = tf.math.add(objective, tf.math.multiply(+0.1, total_batches))
            #objective_weight = 1.0# Adjust as needed
            #objective = -objective_weight * total_sampled_bits_per_event * batch_size

            # Total loss
            loss = objective + penalties

        # Compute gradients
        #print(batch_size.numpy(), sample_size.numpy(), float(total_allocated_bits_in_batch.numpy())/(1024. * 1024. * 8.))
        gradients = tape.gradient(loss, [batch_size, sample_size, total_batches])
        optimizer.apply_gradients(zip(gradients, [batch_size, sample_size, total_batches]))

        # Apply variable bounds
        batch_size.assign(tf.clip_by_value(batch_size, 1.0, batch_size_max))
        sample_size.assign(tf.clip_by_value(sample_size, 1.0, sample_size_max))
        total_batches.assign(tf.clip_by_value(total_batches, 1.0, total_batches_max))
        #print(objective, total_sampled_bits_per_event, penalties)
        abs_loss = abs(loss.numpy())
        diff = abs(prev_loss - abs_loss)
        fractional_change = float(diff)/(prev_loss + 1.)
        if fractional_change <= tolerance:
            wait_time += 1
            if wait_time >= patience:
                print(f"Early stopping at iteration {iteration}")
                break
        else:
            wait_time = 0  # Reset wait counter if loss improves
        prev_loss = abs_loss
        #print(prev_loss, abs_loss, wait_time, tolerance, diff, float(diff)/prev_loss)
    # Round variables to nearest integers
    batch_size_opt = int(round(batch_size.numpy()))
    sample_size_opt = int(round(sample_size.numpy()))
    total_batches_opt = int(round(total_batches.numpy()))
    total_sampled_bits_per_event_opt = total_batches_opt * batch_size_opt * sample_size_opt * int(bits_in_bitpack_dtype.numpy())
    bitpack_allocated_bits_in_batch_opt = batch_size_opt * int(num_events.numpy()) * sample_size_opt * int(bits_in_bitpack_dtype.numpy())
    sampler_allocated_bits_in_batch_opt = batch_size_opt * int(num_events.numpy()) * sample_size_opt * int(bits_in_sampler_dtype.numpy())
    total_allocated_bits_in_batch_opt = bitpack_allocated_bits_in_batch_opt + sampler_allocated_bits_in_batch_opt

    # Prepare result
    result = {
        'batch_size': batch_size_opt,
        'sample_size': sample_size_opt,
        'total_batches': total_batches_opt,
        'total_sampled_bits_per_event': total_sampled_bits_per_event_opt,
        'bitpack_allocated_bits_in_batch': bitpack_allocated_bits_in_batch_opt,
        'sampler_allocated_bits_in_batch': sampler_allocated_bits_in_batch_opt,
        'total_allocated_bits_in_batch': total_allocated_bits_in_batch_opt,
        'ObjectiveValue': total_sampled_bits_per_event_opt,
        'num_events': int(num_events.numpy()),
        'bits_in_sampler_dtype': int(bits_in_sampler_dtype.numpy()),
        'bits_in_bitpack_dtype': int(bits_in_bitpack_dtype.numpy()),
    }

    # Print results
    print(f"num_events                      : {result['num_events']}")
    print(f"total_sampled_bits_per_event    : {result['total_sampled_bits_per_event']} bits")
    print("---------------------------------------------------------")
    print(f"bits_in_sampler_dtype           : {result['bits_in_sampler_dtype']} bits")
    print(f"bits_in_bitpack_dtype           : {result['bits_in_bitpack_dtype']} bits")
    print("---------------------------------------------------------")
    print(f"sample_size                     : {result['sample_size']}")
    print(f"batch_size                      : {result['batch_size']}")
    print(f"total_batches                   : {result['total_batches']}")
    print("---------------------------------------------------------")
    print(f"total_allocated_bytes_in_batch  : {result['total_allocated_bits_in_batch'] // (1024 * 1024 * 8)} Mbyte")
    print(f"sampler_allocated_bits_in_batch : {result['sampler_allocated_bits_in_batch']} bits")
    print(f"bitpack_allocated_bits_in_batch : {result['bitpack_allocated_bits_in_batch']} bits")
    print("---------------------------------------------------------")

    return result

def main():
    # Define constants/parameters
    num_events = 2 ** 20  # Example: 67,108,864 events
    max_bytes = int(1.5 * 2 ** 32)  # 1.5 times 4 GiB
    batch_size_range = (1, None)  # Minimum batch size is 2
    sample_size_range = (1, None)  # Minimum sample size is 2
    total_batches_range = (1, 128)  # No maximum on total batches

    # Run optimization
    result = optimize_batch_and_sample_size(
        num_events=num_events,
        max_bytes=max_bytes,
        sampled_bits_per_event_range=(2**20, None),
        sampler_dtype=tf.float64,
        bitpack_dtype=tf.uint64,
        batch_size_range=batch_size_range,
        sample_size_range=sample_size_range,
        total_batches_range=total_batches_range,
        learning_rate=1.0,
        max_iterations=1000,
        tolerance=1e-8,
    )


if __name__ == "__main__":
    main()