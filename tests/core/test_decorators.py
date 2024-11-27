import unittest
from io import StringIO

from pracciolini.core.grammar import Grammar
from pracciolini.core.registry import GrammarRegistry
from pracciolini.core.decorators import translation, validation, load, save


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


# Define a validation using the decorator
@validation("English")
def validate_english_text(text) -> bool:
    # Simple validation: check that text contains only English letters and spaces
    return all(c.isalpha() or c.isspace() for c in text)


class TestValidations(unittest.TestCase):
    def test_validate_english_text(self):
        # Test the validation function directly
        self.assertTrue(validate_english_text("Hello World"))
        self.assertFalse(validate_english_text("Hello World!"))  # Exclamation mark is invalid

        # Test the validation through the registry
        grammar = Grammar("English")
        validations = GrammarRegistry.get_validations(grammar)
        self.assertTrue(any(v.key == "validate_english_text(English)" for v in validations))
        # Find our validation
        validation_func = None
        for v in validations:
            if v.key == "validate_english_text(English)":
                validation_func = v.validate_func
                break
        self.assertIsNotNone(validation_func)
        self.assertTrue(validation_func("Hello World"))
        self.assertFalse(validation_func("Hello World!"))


# Define a loader using the decorator
@load("English", "txt")
def load_english_text(file_like) -> str:
    # For the test, assume file_like is a file-like object
    return file_like.read()


class TestLoaders(unittest.TestCase):
    def test_load_english_text(self):
        # Test the loader function directly
        file_content = "Hello World"
        file_like = StringIO(file_content)
        self.assertEqual(load_english_text(file_like), "Hello World")

        # Test the loader through the registry
        grammar = Grammar("English")
        loaders = GrammarRegistry.get_loaders(grammar, "txt")
        self.assertTrue(any(loader.key == "load_english_text(English, .txt)" for loader in loaders))
        # Find our loader
        loader_func = None
        for loader in loaders:
            if loader.key == "load_english_text(English, .txt)":
                loader_func = loader.load_func
                break
        self.assertIsNotNone(loader_func)
        file_like = StringIO(file_content)
        self.assertEqual(loader_func(file_like), "Hello World")


# Define a saver using the decorator
@save("English", "txt")
def save_english_text(text, file_like) -> None:
    # Assume file_like is a file-like object
    file_like.write(text)


class TestSavers(unittest.TestCase):
    def test_save_english_text(self):
        # Test the saver function directly
        file_like = StringIO()
        save_english_text("Hello World", file_like)
        self.assertEqual(file_like.getvalue(), "Hello World")

        # Test the saver through the registry
        grammar = Grammar("English")
        savers = GrammarRegistry.get_savers(grammar, "txt")
        self.assertTrue(any(saver.key == "save_english_text(English, .txt)" for saver in savers))
        # Find our saver
        saver_func = None
        for saver in savers:
            if saver.key == "save_english_text(English, .txt)":
                saver_func = saver.save_func
                break
        self.assertIsNotNone(saver_func)
        file_like = StringIO()
        saver_func("Hello World", file_like)
        self.assertEqual(file_like.getvalue(), "Hello World")

if __name__ == "__main__":
    unittest.main()
