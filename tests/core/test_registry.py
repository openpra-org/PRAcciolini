import unittest
from io import StringIO

from pracciolini.core.grammar import Grammar
from pracciolini.core.registry import GrammarRegistry
from pracciolini.core.translation import Translation
from pracciolini.core.validation import Validation
from pracciolini.core.loading import Loader
from pracciolini.core.saving import Saver


class TestGrammarRegistry(unittest.TestCase):
    def setUp(self):
        """Setup a clean registry for each test."""
        # Clear the graph and all mappings
        GrammarRegistry.graph.clear()
        GrammarRegistry.validations.clear()
        GrammarRegistry.loaders.clear()
        GrammarRegistry.savers.clear()

        # Define grammars
        self.g1 = Grammar("English")
        self.g2 = Grammar("French")
        self.g3 = Grammar("Spanish")

        # Define translations
        self.t1 = Translation("basic", lambda text: text.replace("hello", "bonjour"))
        self.t2 = Translation("formal", lambda text: text.replace("hello", "salut"))
        self.t3 = Translation("casual", lambda text: text.replace("hello", "hola"))

        # Define validations
        self.v1 = Validation("length_check", lambda text: len(text) > 0)
        self.v2 = Validation("no_numbers", lambda text: all(not c.isdigit() for c in text))

        # Define loaders
        self.l1 = Loader("load_txt", lambda file_like: file_like.read())
        self.l2 = Loader("load_uppercase", lambda file_like: file_like.read().upper())

        # Define savers
        self.s1 = Saver("save_txt", lambda data, file_like: file_like.write(data))
        self.s2 = Saver("save_lowercase", lambda data, file_like: file_like.write(data.lower()))

    # Existing translation tests ...

    def test_add_validation(self):
        """Test adding validations to the registry."""
        GrammarRegistry.add_validation(self.g1, self.v1)
        self.assertIn(self.g1, GrammarRegistry.validations)
        self.assertIn(self.v1, GrammarRegistry.validations[self.g1])

    def test_get_validations(self):
        """Test retrieving validations for a grammar."""
        GrammarRegistry.add_validation(self.g1, self.v1)
        GrammarRegistry.add_validation(self.g1, self.v2)
        validations = GrammarRegistry.get_validations(self.g1)
        self.assertEqual(len(validations), 2)
        self.assertIn(self.v1, validations)
        self.assertIn(self.v2, validations)

    def test_validation_functionality(self):
        """Test the validation functions."""
        GrammarRegistry.add_validation(self.g1, self.v1)
        GrammarRegistry.add_validation(self.g1, self.v2)
        validations = GrammarRegistry.get_validations(self.g1)
        text = "Hello World"
        for validation in validations:
            result = validation.validate_func(text)
            self.assertTrue(result)
        invalid_text = "Hello World 123"
        # The 'no_numbers' validation should fail
        for validation in validations:
            result = validation.validate_func(invalid_text)
            if validation.key == "no_numbers":
                self.assertFalse(result)
            else:
                self.assertTrue(result)

    def test_add_loader(self):
        """Test adding loaders to the registry."""
        GrammarRegistry.add_loader(self.g1, "txt", self.l1)
        self.assertIn(self.g1, GrammarRegistry.loaders)
        self.assertIn("txt", GrammarRegistry.loaders[self.g1])
        self.assertIn(self.l1, GrammarRegistry.loaders[self.g1]["txt"])

    def test_get_loaders(self):
        """Test retrieving loaders for a grammar and extension."""
        GrammarRegistry.add_loader(self.g1, "txt", self.l1)
        GrammarRegistry.add_loader(self.g1, "txt", self.l2)
        loaders = GrammarRegistry.get_loaders(self.g1, "txt")
        self.assertEqual(len(loaders), 2)
        self.assertIn(self.l1, loaders)
        self.assertIn(self.l2, loaders)

    def test_loader_functionality(self):
        """Test the loader functions."""
        GrammarRegistry.add_loader(self.g1, "txt", self.l1)
        GrammarRegistry.add_loader(self.g1, "txt", self.l2)
        loaders = GrammarRegistry.get_loaders(self.g1, "txt")
        file_content = "Hello World"
        file_like = StringIO(file_content)
        # Test each loader
        for loader in loaders:
            file_like.seek(0)  # Reset the file-like object
            result = loader.load_func(file_like)
            if loader.key == "load_txt":
                self.assertEqual(result, "Hello World")
            elif loader.key == "load_uppercase":
                self.assertEqual(result, "HELLO WORLD")

    def test_add_saver(self):
        """Test adding savers to the registry."""
        GrammarRegistry.add_saver(self.g1, "txt", self.s1)
        self.assertIn(self.g1, GrammarRegistry.savers)
        self.assertIn("txt", GrammarRegistry.savers[self.g1])
        self.assertIn(self.s1, GrammarRegistry.savers[self.g1]["txt"])

    def test_get_savers(self):
        """Test retrieving savers for a grammar and extension."""
        GrammarRegistry.add_saver(self.g1, "txt", self.s1)
        GrammarRegistry.add_saver(self.g1, "txt", self.s2)
        savers = GrammarRegistry.get_savers(self.g1, "txt")
        self.assertEqual(len(savers), 2)
        self.assertIn(self.s1, savers)
        self.assertIn(self.s2, savers)

    def test_saver_functionality(self):
        """Test the saver functions."""
        GrammarRegistry.add_saver(self.g1, "txt", self.s1)
        GrammarRegistry.add_saver(self.g1, "txt", self.s2)
        savers = GrammarRegistry.get_savers(self.g1, "txt")
        data = "Hello World"
        # Test each saver
        for saver in savers:
            file_like = StringIO()
            saver.save_func(data, file_like)
            if saver.key == "save_txt":
                self.assertEqual(file_like.getvalue(), "Hello World")
            elif saver.key == "save_lowercase":
                self.assertEqual(file_like.getvalue(), "hello world")

    def test_loader_nonexistent_extension(self):
        """Test retrieving loaders for a nonexistent extension."""
        loaders = GrammarRegistry.get_loaders(self.g1, "json")
        self.assertEqual(loaders, [])

    def test_saver_nonexistent_extension(self):
        """Test retrieving savers for a nonexistent extension."""
        savers = GrammarRegistry.get_savers(self.g1, "json")
        self.assertEqual(savers, [])

    def test_validation_nonexistent_grammar(self):
        """Test retrieving validations for a grammar with no validations."""
        validations = GrammarRegistry.get_validations(self.g2)
        self.assertEqual(validations, [])

    # Existing translation tests ...

    def test_add_grammar(self):
        """Test adding grammars to the registry."""
        GrammarRegistry.add_grammar(self.g1)
        self.assertIn(self.g1, GrammarRegistry.graph.nodes)

    def test_add_duplicate_grammar(self):
        """Test adding duplicate grammars to the registry. Adding duplicate nodes does nothing interesting. """
        GrammarRegistry.add_grammar(self.g1)
        g1_duplicate = Grammar("English")
        GrammarRegistry.add_grammar(g1_duplicate)
        # Since Grammar uses 'key' for equality, both should be considered the same node
        self.assertIn(self.g1, GrammarRegistry.graph.nodes)
        self.assertIn(g1_duplicate, GrammarRegistry.graph.nodes)
        self.assertEqual(len(GrammarRegistry.graph.nodes), 1)

    def test_add_translation(self):
        """Test adding translations to the registry."""
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        # Check for the presence of the edge with the correct key
        self.assertIn((self.g1, self.g2, 'basic'), GrammarRegistry.graph.edges)
        # Additionally, check if the translation object is correctly associated
        self.assertEqual(GrammarRegistry.graph[self.g1][self.g2]['basic']['translation'], self.t1)

    def test_get_translations(self):
        """Test retrieving translations between two grammars."""
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        translations = GrammarRegistry.get_translations(self.g1, self.g2)
        self.assertIn('basic', translations)
        self.assertEqual(translations['basic']['translation'], self.t1)

    def test_translate(self):
        """Test the translate function."""
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        result = GrammarRegistry.translate(self.g1, self.g2, "basic", "hello world")
        self.assertEqual(result, "bonjour world")

    def test_translate_nonexistent_translation(self):
        """Test translating with a nonexistent translation key."""
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        with self.assertRaises(ValueError):
            GrammarRegistry.translate(self.g1, self.g2, "nonexistent", "hello world")

    def test_multiple_translations(self):
        """Test handling multiple translations between the same grammars."""
        GrammarRegistry.add_translation(self.g1, self.g2, self.t1)
        GrammarRegistry.add_translation(self.g1, self.g2, self.t2)
        result1 = GrammarRegistry.translate(self.g1, self.g2, "basic", "hello world")
        result2 = GrammarRegistry.translate(self.g1, self.g2, "formal", "hello world")
        self.assertEqual(result1, "bonjour world")
        self.assertEqual(result2, "salut world")


if __name__ == "__main__":
    unittest.main()
