import unittest

import tensorflow as tf

from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph

class SubGraphConstructionTests(unittest.TestCase):
    def setUp(self):
        # might need pop count
        samples_a = [0b00000000, 0b00000001, 0b00000010]
        samples_b = [0b00000100, 0b00001000, 0b00010000]
        samples_c = [0b00100000, 0b01000000, 0b10000000]
        samples_d = [0b11111111, 0b11111111, 0b11111111]
        self.A = Tensor(tf.constant(samples_a, dtype=tf.uint8), name="samples_a", shape=[1, None])
        self.B = Tensor(tf.constant(samples_b, dtype=tf.uint8), name="samples_b", shape=[1, None])
        self.C = Tensor(tf.constant(samples_c, dtype=tf.uint8), name="samples_c", shape=[1, None])
        self.D = Tensor(tf.constant(samples_d, dtype=tf.uint8), name="samples_d", shape=[1, None])
        self.concat_ABC = Tensor(tf.constant([samples_a, samples_b, samples_c], dtype=tf.uint8), name="concat_abc")

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
        subgraph_concat = SubGraph(name="concat")
        subgraph_concat.register_input(self.concat_ABC)
        subgraph_concat.register_input(self.D)
        output_2d = subgraph_concat.bitwise_and(self.concat_ABC, self.D, name="2d_(A^B)")
        subgraph_concat.register_output(output_2d)
        results_2d = subgraph_concat.execute_function()()[0].numpy()
        binary_literals_2d = [f'0b{result:08b}' for result in results_2d]
        print(f"binary_literals_2d: {binary_literals_2d}")