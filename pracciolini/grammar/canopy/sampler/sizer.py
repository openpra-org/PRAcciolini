import tensorflow as tf

def compute_bits_in_dtype(tensor_type: tf.DType) -> int:
    """
    Computes the number of bits in the given data type.

    Args:
        tensor_type (tf.DType): The tensor data type.

    Returns:
        int: Number of bits in the data type.
    """
    return tf.dtypes.as_dtype(tensor_type).size * 8

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
        sampled_bits_per_event_range (tuple): Desired cumulative bits per event (min, max).
        sampler_dtype (tf.DType): Data type for event data (e.g., tf.float32).
        bitpack_dtype (tf.DType): Data type for bit-packing (e.g., tf.uint8).
        batch_size_range (tuple): (batch_size_min, batch_size_max).
        sample_size_range (tuple): (sample_size_min, sample_size_max).
        total_batches_range (tuple): (total_batches_min, total_batches_max).
        learning_rate (float): Learning rate for the optimizer.
        max_iterations (int): Maximum number of iterations for the optimizer.
        tolerance (float): Tolerance for stopping criteria.

    Returns:
        dict: A dictionary containing tensors of the optimal values and related computations.
    """
    # Compute bits in data types (Python integers)
    bits_in_sampler_dtype = compute_bits_in_dtype(sampler_dtype)
    bits_in_bitpack_dtype = compute_bits_in_dtype(bitpack_dtype)

    # Variable bounds (Python scalars)
    sampled_bits_per_event_min = bits_in_bitpack_dtype if sampled_bits_per_event_range[0] is None else sampled_bits_per_event_range[0]
    sampled_bits_per_event_max = 2**31 - 1 if sampled_bits_per_event_range[1] is None else sampled_bits_per_event_range[1]

    batch_size_min = batch_size_range[0]
    batch_size_max = batch_size_range[1] if batch_size_range[1] is not None else sampled_bits_per_event_max // bits_in_bitpack_dtype

    sample_size_min = sample_size_range[0]
    sample_size_max = sample_size_range[1] if sample_size_range[1] is not None else sampled_bits_per_event_max // bits_in_bitpack_dtype

    total_batches_min = total_batches_range[0]
    total_batches_max = total_batches_range[1] if total_batches_range[1] is not None else sampled_bits_per_event_max // bits_in_bitpack_dtype

    # Convert inputs to TensorFlow constants
    num_events = tf.constant(num_events, dtype=tf.float32)
    max_bits = tf.constant(max_bytes * 8, dtype=tf.float32)  # Convert max_bytes to bits

    bits_in_sampler_dtype = tf.constant(bits_in_sampler_dtype, dtype=tf.float32)
    bits_in_bitpack_dtype = tf.constant(bits_in_bitpack_dtype, dtype=tf.float32)

    sampled_bits_per_event_min = tf.constant(sampled_bits_per_event_min, dtype=tf.float32)
    sampled_bits_per_event_max = tf.constant(sampled_bits_per_event_max, dtype=tf.float32)

    batch_size_min = tf.constant(batch_size_min, dtype=tf.float32)
    batch_size_max = tf.constant(batch_size_max, dtype=tf.float32)
    sample_size_min = tf.constant(sample_size_min, dtype=tf.float32)
    sample_size_max = tf.constant(sample_size_max, dtype=tf.float32)
    total_batches_min = tf.constant(total_batches_min, dtype=tf.float32)
    total_batches_max = tf.constant(total_batches_max, dtype=tf.float32)

    # Initialize decision variables as TensorFlow variables with initial guesses
    batch_size = tf.Variable(batch_size_min, trainable=True, dtype=tf.float32, name='batch_size')
    sample_size = tf.Variable(sample_size_min, trainable=True, dtype=tf.float32, name='sample_size')
    total_batches = tf.Variable(total_batches_min, trainable=True, dtype=tf.float32, name='total_batches')

    # Use GradientTape for automatic differentiation
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.8)

    @tf.function
    def bounds_penalty(x_min_, x_, x_max_):
        return tf.square(tf.nn.relu(x_min_ - x_)) + tf.square(tf.nn.relu(x_ - x_max_))

    @tf.function
    def compute_forward_pass(total_batches_, batch_size_, sample_size_, num_events_, bits_in_bitpack_dtype_, bits_in_sampler_dtype_):
        ## Memory utilization related stats
        num_float_samples_in_batch_ = batch_size_ * num_events_ * sample_size_ * bits_in_bitpack_dtype_
        sampler_allocated_bits_in_batch_ = num_float_samples_in_batch_ * bits_in_sampler_dtype_
        bitpack_allocated_bits_in_batch_ = batch_size_ * num_events_ * sample_size_ * bits_in_bitpack_dtype_
        total_allocated_bits_in_batch_ = sampler_allocated_bits_in_batch_ + bitpack_allocated_bits_in_batch_
        ## Event related stats
        total_sampled_bits_per_event_ = total_batches_ * batch_size_ * sample_size_ * bits_in_bitpack_dtype_
        return (total_sampled_bits_per_event_, total_allocated_bits_in_batch_,
                num_float_samples_in_batch_, sampler_allocated_bits_in_batch_,
                bitpack_allocated_bits_in_batch_)

    for _ in tf.range(max_iterations):
        with tf.GradientTape() as tape:
            # Compute forward pass
            (total_sampled_bits_per_event,
             total_allocated_bits_in_batch, _, _, _) = compute_forward_pass(
                total_batches, batch_size, sample_size, num_events,
                bits_in_bitpack_dtype, bits_in_sampler_dtype)

            # Compute penalties
            memory_size_penalty = bounds_penalty(1.0, total_allocated_bits_in_batch, max_bits)
            batch_size_penalty = bounds_penalty(batch_size_min, batch_size, batch_size_max)
            sample_size_penalty = bounds_penalty(sample_size_min, sample_size, sample_size_max)
            total_batches_penalty = bounds_penalty(total_batches_min, total_batches, total_batches_max)
            total_sampled_bits_penalty = bounds_penalty(sampled_bits_per_event_min, total_sampled_bits_per_event, sampled_bits_per_event_max)
            penalties = tf.reduce_sum([
                memory_size_penalty, batch_size_penalty, sample_size_penalty,
                total_batches_penalty, total_sampled_bits_penalty])

            # Objective function
            # Maximize total_sampled_bits_per_event and minimize total_batches
            objective = -total_sampled_bits_per_event + 0.1 * total_batches

            # Total loss
            loss = objective + penalties

        # Compute gradients
        gradients = tape.gradient(loss, [batch_size, sample_size, total_batches])
        optimizer.apply_gradients(zip(gradients, [batch_size, sample_size, total_batches]))

        # Apply variable bounds
        batch_size.assign(tf.clip_by_value(batch_size, batch_size_min, batch_size_max))
        sample_size.assign(tf.clip_by_value(sample_size, sample_size_min, sample_size_max))
        total_batches.assign(tf.clip_by_value(total_batches, total_batches_min, total_batches_max))

    # Final values
    batch_size_final = tf.cast(tf.round(batch_size), dtype=tf.float32)
    sample_size_final = tf.cast(tf.round(sample_size), dtype=tf.float32)
    total_batches_final = tf.cast(tf.round(total_batches), dtype=tf.float32)

    # Compute final metrics
    (total_sampled_bits_per_event__,
     total_allocated_bits_in_batch__,
     num_float_samples_in_batch__,
     sampler_allocated_bits_in_batch__,
     bitpack_allocated_bits_in_batch__) = compute_forward_pass(
        total_batches_final, batch_size_final, sample_size_final, num_events,
        bits_in_bitpack_dtype, bits_in_sampler_dtype)

    # Prepare result (conversion outside tf.function scope)
    result = {
        'num_events': num_events,
        'batch_size': batch_size_final,
        'sample_size': sample_size_final,
        'total_batches': total_batches_final,
        'total_sampled_bits_per_event': total_sampled_bits_per_event__,
        'bitpack_allocated_bits_in_batch': bitpack_allocated_bits_in_batch__,
        'sampler_allocated_bits_in_batch': sampler_allocated_bits_in_batch__,
        'total_allocated_bits_in_batch': total_allocated_bits_in_batch__,
        'num_samples_in_batch': num_float_samples_in_batch__,
        'bits_in_sampler_dtype': bits_in_sampler_dtype,
        'bits_in_bitpack_dtype': bits_in_bitpack_dtype,
    }
    return result

def build_result_dictionary(result_tensors):
    """
    Converts tensors in the result dictionary to Python native types.

    Args:
        result_tensors (dict): A dictionary containing tensors.

    Returns:
        dict: A dictionary with tensors converted to Python native types.
    """
    result = {}
    for key, value in result_tensors.items():
        # Check if the tensor is a scalar tensor
        if isinstance(value, tf.Tensor) and value.shape == ():
            result[key] = value.numpy().item()  # Convert scalar tensor to Python scalar
        else:
            result[key] = value.numpy()  # Convert tensor to NumPy array
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
    print(f"num_float_samples_in_batch      : {result['num_samples_in_batch']}")
    print(f"total_allocated_bytes_in_batch  : {result['total_allocated_bits_in_batch'] // (1024 * 1024 * 8)} Mbyte")
    print(f"sampler_allocated_bits_in_batch : {result['sampler_allocated_bits_in_batch']} bits")
    print(f"bitpack_allocated_bits_in_batch : {result['bitpack_allocated_bits_in_batch']} bits")
    print("---------------------------------------------------------")
    return result

def main():
    # Define constants/parameters
    num_events = 2 ** 10  # Example: 67,108,864 events
    max_bytes = int(1.5 * 2 ** 32)  # 1.5 times 4 GiB
    batch_size_range = (1, None)  # Minimum batch size is 2
    sample_size_range = (1, None)  # Minimum sample size is 2
    total_batches_range = (1, None)  # No maximum on total batches

    # Run optimization
    result = optimize_batch_and_sample_size(
        num_events=num_events,
        max_bytes=max_bytes,
        sampled_bits_per_event_range=(2**20, 2**21),
        sampler_dtype=tf.float32,
        bitpack_dtype=tf.uint8,
        batch_size_range=batch_size_range,
        sample_size_range=sample_size_range,
        total_batches_range=total_batches_range,
        learning_rate=1.0,
        max_iterations=1000,
        tolerance=1e-8,
    )


if __name__ == "__main__":
    main()