import unittest

from pracciolini.core.grammar import Grammar
from pracciolini.core.registry import GrammarRegistry
from pracciolini.core.decorators import translation


# Define translations using the decorator
@translation("English", "Spanish")
def greet_in_spanish(text) -> str:
    return text.replace("Hello", "Hola")


@translation("English", "German")
def greet_in_german(text) -> str:
    return text.replace("Hello", "Hallo")


class TestTranslations(unittest.TestCase):
    def test_greet_in_spanish(self):
        # Test the translation directly
        self.assertEqual(greet_in_spanish("Hello World"), "Hola World")

        # Test the translation through the registry
        result = GrammarRegistry.translate(Grammar("English"), Grammar("Spanish"),
                                           "greet_in_spanish(English) -> Spanish", "Hello World")
        self.assertEqual(result, "Hola World")

    def test_greet_in_german(self):
        # Test the translation directly
        self.assertEqual(greet_in_german("Hello World"), "Hallo World")

        # Test the translation through the registry
        result = GrammarRegistry.translate(Grammar("English"), Grammar("German"), "greet_in_german(English) -> German",
                                           "Hello World")
        self.assertEqual(result, "Hallo World")

    def test_translation_not_found(self):
        # Test for a non-existent translation
        with self.assertRaises(ValueError):
            GrammarRegistry.translate(Grammar("English"), Grammar("French"), "non_existent_translation", "Hello World")


if __name__ == "__main__":
    unittest.main()
