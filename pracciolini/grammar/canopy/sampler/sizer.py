import tensorflow as tf

def compute_bits_in_dtype(tensor_type: tf.DType) -> int:
    """
    Computes the number of bits in the given data type.
    """
    return tf.dtypes.as_dtype(tensor_type).size * 8

class BatchSampleSizeOptimizer(tf.Module):
    def __init__(
        self,
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
        super().__init__()
        # Compute bits in data types
        self.bits_in_sampler_dtype = compute_bits_in_dtype(sampler_dtype)
        self.bits_in_bitpack_dtype = compute_bits_in_dtype(bitpack_dtype)

        # Variable bounds
        self.sampled_bits_per_event_min = self.bits_in_bitpack_dtype if sampled_bits_per_event_range[0] is None else sampled_bits_per_event_range[0]
        self.sampled_bits_per_event_max = 2**31 - 1 if sampled_bits_per_event_range[1] is None else sampled_bits_per_event_range[1]

        self.batch_size_min = batch_size_range[0]
        self.batch_size_max = batch_size_range[1] if batch_size_range[1] is not None else self.sampled_bits_per_event_max // self.bits_in_bitpack_dtype

        self.sample_size_min = sample_size_range[0]
        self.sample_size_max = sample_size_range[1] if sample_size_range[1] is not None else self.sampled_bits_per_event_max // self.bits_in_bitpack_dtype

        self.total_batches_min = total_batches_range[0]
        self.total_batches_max = total_batches_range[1] if total_batches_range[1] is not None else self.sampled_bits_per_event_max // self.bits_in_bitpack_dtype

        # Initialize variables
        self.num_events = tf.constant(num_events, dtype=tf.float32)
        self.max_bits = tf.constant(max_bytes * 8, dtype=tf.float32)  # Convert max_bytes to bits

        self.bits_in_sampler_dtype = tf.constant(self.bits_in_sampler_dtype, dtype=tf.float32)
        self.bits_in_bitpack_dtype = tf.constant(self.bits_in_bitpack_dtype, dtype=tf.float32)

        self.sampled_bits_per_event_min = tf.constant(self.sampled_bits_per_event_min, dtype=tf.float32)
        self.sampled_bits_per_event_max = tf.constant(self.sampled_bits_per_event_max, dtype=tf.float32)

        self.batch_size_min = tf.constant(self.batch_size_min, dtype=tf.float32)
        self.batch_size_max = tf.constant(self.batch_size_max, dtype=tf.float32)
        self.sample_size_min = tf.constant(self.sample_size_min, dtype=tf.float32)
        self.sample_size_max = tf.constant(self.sample_size_max, dtype=tf.float32)
        self.total_batches_min = tf.constant(self.total_batches_min, dtype=tf.float32)
        self.total_batches_max = tf.constant(self.total_batches_max, dtype=tf.float32)

        # Initialize decision variables as TensorFlow variables
        self.batch_size = tf.Variable(self.batch_size_min, trainable=True, dtype=tf.float32, name='batch_size')
        self.sample_size = tf.Variable(self.sample_size_min, trainable=True, dtype=tf.float32, name='sample_size')
        self.total_batches = tf.Variable(self.total_batches_min, trainable=True, dtype=tf.float32, name='total_batches')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.8)
        self.max_iterations = tf.constant(max_iterations)
        self.tolerance = tf.constant(tolerance)
        print(self.max_iterations)

    @tf.function
    def bounds_penalty(self, x_min_, x_, x_max_):
        return tf.square(tf.nn.relu(x_min_ - x_)) + tf.square(tf.nn.relu(x_ - x_max_))

    @tf.function
    def compute_forward_pass(self, total_batches_, batch_size_, sample_size_, num_events_, bits_in_bitpack_dtype_, bits_in_sampler_dtype_):
        # Memory utilization related stats
        num_float_samples_in_batch_ = batch_size_ * num_events_ * sample_size_ * bits_in_bitpack_dtype_
        sampler_allocated_bits_in_batch_ = num_float_samples_in_batch_ * bits_in_sampler_dtype_
        bitpack_allocated_bits_in_batch_ = batch_size_ * num_events_ * sample_size_ * bits_in_bitpack_dtype_
        total_allocated_bits_in_batch_ = sampler_allocated_bits_in_batch_ + bitpack_allocated_bits_in_batch_
        # Event related stats
        total_sampled_bits_per_event_ = total_batches_ * batch_size_ * sample_size_ * bits_in_bitpack_dtype_
        return (total_sampled_bits_per_event_, total_allocated_bits_in_batch_,
                num_float_samples_in_batch_, sampler_allocated_bits_in_batch_,
                bitpack_allocated_bits_in_batch_)

    @tf.function
    def optimize(self):
        for _ in tf.range(self.max_iterations):
            with tf.GradientTape() as tape:
                # Compute forward pass
                (total_sampled_bits_per_event,
                 total_allocated_bits_in_batch, _, _, _) = self.compute_forward_pass(
                    self.total_batches, self.batch_size, self.sample_size, self.num_events,
                    self.bits_in_bitpack_dtype, self.bits_in_sampler_dtype)

                # Compute penalties
                memory_size_penalty = self.bounds_penalty(1.0, total_allocated_bits_in_batch, self.max_bits)
                batch_size_penalty = self.bounds_penalty(self.batch_size_min, self.batch_size, self.batch_size_max)
                sample_size_penalty = self.bounds_penalty(self.sample_size_min, self.sample_size, self.sample_size_max)
                total_batches_penalty = self.bounds_penalty(self.total_batches_min, self.total_batches, self.total_batches_max)
                total_sampled_bits_penalty = self.bounds_penalty(self.sampled_bits_per_event_min, total_sampled_bits_per_event, self.sampled_bits_per_event_max)
                penalties = tf.reduce_sum([
                    memory_size_penalty, batch_size_penalty, sample_size_penalty,
                    total_batches_penalty, total_sampled_bits_penalty])

                # Objective function
                # Maximize total_sampled_bits_per_event and minimize total_batches
                objective = -total_sampled_bits_per_event + 0.1 * self.total_batches

                # Total loss
                loss = objective + penalties

            # Compute gradients
            gradients = tape.gradient(loss, [self.batch_size, self.sample_size, self.total_batches])
            self.optimizer.apply_gradients(zip(gradients, [self.batch_size, self.sample_size, self.total_batches]))

            # Apply variable bounds
            self.batch_size.assign(tf.clip_by_value(self.batch_size, self.batch_size_min, self.batch_size_max))
            self.sample_size.assign(tf.clip_by_value(self.sample_size, self.sample_size_min, self.sample_size_max))
            self.total_batches.assign(tf.clip_by_value(self.total_batches, self.total_batches_min, self.total_batches_max))

    def get_results(self):
        # Final values
        batch_size_final = tf.cast(tf.round(self.batch_size), dtype=tf.float32)
        sample_size_final = tf.cast(tf.round(self.sample_size), dtype=tf.float32)
        total_batches_final = tf.cast(tf.round(self.total_batches), dtype=tf.float32)

        # Compute final metrics
        (total_sampled_bits_per_event__,
         total_allocated_bits_in_batch__,
         num_float_samples_in_batch__,
         sampler_allocated_bits_in_batch__,
         bitpack_allocated_bits_in_batch__) = self.compute_forward_pass(
            total_batches_final, batch_size_final, sample_size_final, self.num_events,
            self.bits_in_bitpack_dtype, self.bits_in_sampler_dtype)

        # Prepare result (conversion outside tf.function scope)
        result = {
            'num_events': self.num_events,
            'batch_size': batch_size_final,
            'sample_size': sample_size_final,
            'total_batches': total_batches_final,
            'total_sampled_bits_per_event': total_sampled_bits_per_event__,
            'bitpack_allocated_bits_in_batch': bitpack_allocated_bits_in_batch__,
            'sampler_allocated_bits_in_batch': sampler_allocated_bits_in_batch__,
            'total_allocated_bits_in_batch': total_allocated_bits_in_batch__,
            'num_samples_in_batch': num_float_samples_in_batch__,
            'bits_in_sampler_dtype': self.bits_in_sampler_dtype,
            'bits_in_bitpack_dtype': self.bits_in_bitpack_dtype,
        }
        return build_result_dictionary(result)

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
    optimizer = BatchSampleSizeOptimizer(
        num_events=100,  # Example value
        max_bytes=2**32,  # Example value
        sampled_bits_per_event_range=(8, 1024),  # Example values
        sampler_dtype=tf.float32,
        bitpack_dtype=tf.uint8,
        batch_size_range=(1, 100),
        sample_size_range=(1, 1000),
        total_batches_range=(1, 50),
        learning_rate=1.0,
        max_iterations=500,
        tolerance=1e-4,
    )

    # Run the optimization
    optimizer.optimize()

    # Get the results
    results = optimizer.get_results()

    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")