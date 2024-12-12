import unittest

import tensorflow as tf

from pracciolini.grammar.canopy.model.tensor import Tensor
from pracciolini.grammar.canopy.model.subgraph import SubGraph

class SubGraphConstructionTests(unittest.TestCase):
    def setUp(self):
        # might need pop count
        samples_a = [0b00000000, 0b00000001, 0b00000010] # 8 * 3 = 24 total samples for A, P(A) = (2/24)  = 0.0833
        samples_b = [0b00000100, 0b00001000, 0b00010000] # 8 * 3 = 24 total samples for B, P(B) = (3/24)  = 0.1250
        samples_c = [0b00100100, 0b01000000, 0b10000000] # 8 * 3 = 24 total samples for C, P(C) = (4/24)  = 0.1666
        samples_d = [0b11111111, 0b11111111, 0b11111111] # 8 * 3 = 24 total samples for D, P(D) = (24/24) = 1.0000
        self.A = Tensor(tf.constant(samples_a, dtype=tf.uint8), name="samples_a", shape=[1, None])
        self.B = Tensor(tf.constant(samples_b, dtype=tf.uint8), name="samples_b", shape=[1, None])
        self.C = Tensor(tf.constant(samples_c, dtype=tf.uint8), name="samples_c", shape=[1, None])
        self.D = Tensor(tf.constant(samples_d, dtype=tf.uint8), name="samples_d", shape=[1, None])

    def test_simple_ops(self):
        # f_11 = A|B|C
        # f_12 = ~D
        # f_21 = f_11 & D
        # f    = f_21 ^ f_12
        four_ops_four_operands = SubGraph(name="f = ((A|B|C)&D)^(~D)")
        four_ops_four_operands.register_input(self.A)
        four_ops_four_operands.register_input(self.B)
        four_ops_four_operands.register_input(self.C)
        four_ops_four_operands.register_input(self.D)
        # intermediates
        f_11 = four_ops_four_operands.bitwise_or(self.A, self.B, self.C, name="f_11 = or(A,B,C)")
        f_21 = four_ops_four_operands.bitwise_and(self.D, f_11, name="f_21 = and(D,f_11)")
        f_12 = four_ops_four_operands.bitwise_not(self.D, name="f_21 = not(D)")
        # output
        f = four_ops_four_operands.bitwise_xor(f_21, f_12, name="f = xor(f_21,f_12)")
        four_ops_four_operands.register_output(f)

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

        tf_graph = four_ops_four_operands.to_tensorflow_model()
        tf_graph.save("four_ops_four_operands.h5")

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