import unittest

from pracciolini.core.grammar import Grammar
from pracciolini.core.registry import GrammarRegistry
from pracciolini.core.translation import Translation


class TestGrammarRegistry(unittest.TestCase):
    def setUp(self):
        """Setup a clean graph for each test."""
        GrammarRegistry.graph.clear()
        self.g1 = Grammar("English")
        self.g2 = Grammar("French")
        self.g3 = Grammar("Spanish")
        self.t1 = Translation("basic", lambda text: text.replace("hello", "bonjour"))
        self.t2 = Translation("formal", lambda text: text.replace("hello", "salut"))
        self.t3 = Translation("casual", lambda text: text.replace("hello", "hola"))

    def test_add_grammar(self):
        """Test adding grammars to the registry."""
        GrammarRegistry.add_grammar(self.g1)
        self.assertIn(self.g1, GrammarRegistry.graph.nodes)

    def test_add_duplicate_grammar(self):
        """Test adding duplicate grammars to the registry. adding duplicate nodes does nothing interesting. """
        GrammarRegistry.add_grammar(self.g1)
        g1_duplicate = Grammar("English")
        GrammarRegistry.add_grammar(g1_duplicate)
        self.assertIn(self.g1, GrammarRegistry.graph.nodes)
        self.assertIn(g1_duplicate, GrammarRegistry.graph.nodes)
        self.assertEqual(len(GrammarRegistry.graph.nodes), 1)

    def test_add_translation(self):
        """Test adding translations to the registry."""
        GrammarRegistry.add_grammar(self.g1)
        GrammarRegistry.add_grammar(self.g2)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        # Check for the presence of the edge with the correct key
        self.assertIn((self.g1, self.g2, 'basic'), GrammarRegistry.graph.edges)
        # Additionally, check if the translation object is correctly associated
        self.assertEqual(GrammarRegistry.graph[self.g1][self.g2]['basic']['translation'], self.t1)

    def test_get_translations(self):
        """Test retrieving translations between two grammars."""
        GrammarRegistry.add_grammar(self.g1)
        GrammarRegistry.add_grammar(self.g2)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        translations = GrammarRegistry.get_translations(self.g1, self.g2)
        self.assertIn('basic', translations)
        self.assertEqual(translations['basic']['translation'], self.t1)

    def test_translate(self):
        """Test the translate function."""
        GrammarRegistry.add_grammar(self.g1)
        GrammarRegistry.add_grammar(self.g2)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        result = GrammarRegistry.translate(self.g1, self.g2, "basic", "hello world")
        self.assertEqual(result, "bonjour world")

    def test_translate_nonexistent_translation(self):
        """Test translating with a nonexistent translation key."""
        GrammarRegistry.add_grammar(self.g1)
        GrammarRegistry.add_grammar(self.g2)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        with self.assertRaises(ValueError):
            GrammarRegistry.translate(self.g1, self.g2, "nonexistent", "hello world")

    def test_multiple_translations(self):
        """Test handling multiple translations between the same grammars."""
        GrammarRegistry.add_grammar(self.g1)
        GrammarRegistry.add_grammar(self.g2)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t2)
        result1 = GrammarRegistry.translate(self.g1, self.g2, "basic", "hello world")
        result2 = GrammarRegistry.translate(self.g1, self.g2, "formal", "hello world")
        self.assertEqual(result1, "bonjour world")
        self.assertEqual(result2, "salut world")


if __name__ == "__main__":
    unittest.main()
