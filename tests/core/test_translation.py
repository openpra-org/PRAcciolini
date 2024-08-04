import unittest

from pracciolini.core.translation import Translation


class TestTranslation(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of Translation objects."""
        def translate_func(x):
            return x.upper()
        translation = Translation("uppercase", translate_func)
        self.assertEqual(translation.key, "uppercase")
        self.assertTrue(callable(translation.translate_func))

    def test_hash(self):
        """Test that the hash function returns consistent results for the same object."""
        def translate_func(x):
            return x.upper()
        translation1 = Translation("uppercase", translate_func)
        translation2 = Translation("uppercase", translate_func)
        self.assertEqual(hash(translation1), hash(translation2))

    def test_hash_uniqueness(self):
        """Test that different translations have different hashes."""
        def func1(x):
            return x.upper()
        def func2(x):
            return x.lower()
        translation1 = Translation("uppercase", func1)
        translation2 = Translation("lowercase", func2)
        self.assertNotEqual(hash(translation1), hash(translation2))

    def test_equality(self):
        """Test equality of two Translation objects with the same key and function."""
        def func(x):
            return x.upper()
        translation1 = Translation("uppercase", func)
        translation2 = Translation("uppercase", func)
        self.assertEqual(translation1, translation2)

    def test_inequality(self):
        """Test inequality of two Translation objects with different keys or functions."""
        def func1(x):
            return x.upper()
        def func2(x):
            return x.lower()
        translation1 = Translation("uppercase", func1)
        translation2 = Translation("lowercase", func2)
        self.assertNotEqual(translation1, translation2)

    def test_equality_with_non_translation(self):
        """Test equality comparison between a Translation object and a non-Translation object."""
        def translate_func(x):
            return x.upper()
        translation = Translation("uppercase", translate_func)
        not_a_translation = "uppercase"
        self.assertNotEqual(translation, not_a_translation)

    def test_repr(self):
        """Test the string representation of a Translation object."""
        def translate_func(x):
            return x.upper()
        translation = Translation("uppercase", translate_func)
        self.assertEqual(repr(translation), "Translation(uppercase)")

    def test_translation_function(self):
        """Test the translation function's behavior."""
        def translate_func(x):
            return x.upper()
        translation = Translation("uppercase", translate_func)
        result = translation.translate_func("hello")
        self.assertEqual(result, "HELLO")


if __name__ == "__main__":
    unittest.main()
