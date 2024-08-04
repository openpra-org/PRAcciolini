import unittest

from pracciolini.core.grammar import Grammar


class TestGrammar(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of Grammar objects."""
        grammar = Grammar("G1")
        self.assertEqual(grammar.key, "G1")

    def test_hash(self):
        """Test that the hash function returns consistent results for the same object."""
        grammar1 = Grammar("TypeA")
        grammar2 = Grammar("TypeA")
        self.assertEqual(hash(grammar1), hash(grammar2))

    def test_hash_uniqueness(self):
        """Test that different grammars have different hashes."""
        grammar1 = Grammar("some-grammar")
        grammar2 = Grammar("another-grammar")
        self.assertNotEqual(hash(grammar1), hash(grammar2))

    def test_equality(self):
        """Test equality of two Grammar objects with the same key."""
        grammar1 = Grammar("my-grammar")
        grammar2 = Grammar("my-grammar")
        self.assertEqual(grammar1, grammar2)

    def test_inequality(self):
        """Test inequality of two Grammar objects with different keys."""
        grammar1 = Grammar("some-grammar")
        grammar2 = Grammar("another-grammar")
        self.assertNotEqual(grammar1, grammar2)

    def test_equality_with_non_grammar(self):
        """Test equality comparison between a Grammar object and a non-Grammar object."""
        grammar = Grammar("grammar-name")
        not_a_grammar = "grammar-name"
        self.assertNotEqual(grammar, not_a_grammar)

    def test_repr(self):
        """Test the string representation of a Grammar object."""
        grammar = Grammar("g1")
        self.assertEqual(repr(grammar), "Grammar(g1)")


if __name__ == "__main__":
    unittest.main()
