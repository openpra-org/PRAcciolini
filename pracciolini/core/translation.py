from typing import Callable, Any


class Translation:
    """
    A class representing a translation mechanism with a unique identifier and a translation function.

    Attributes:
        key (str): The unique identifier for the translation.
        translate_func (Callable[[Any], Any]): The function used to perform the translation.

    Methods:
        __hash__: Returns the hash based on the translation's key and function.
        __eq__: Checks equality with another Translation object based on the key and function.
        __repr__: Provides the official string representation of the Translation object.
    """

    def __init__(self, key: str, translate_func: Callable[[Any], Any]):
        """
        Initializes the Translation instance with a specified key and translation function.

        Args:
            key (str): The unique identifier for the translation.
            translate_func (Callable[[Any], Any]): The function that performs the translation.
        """
        self.key = key
        self.translate_func = translate_func

    def __hash__(self) -> int:
        """
        Returns the hash of the Translation instance based on its key and translation function.

        Returns:
            int: The hash value of the translation's key and function.
        """
        return hash((self.key, self.translate_func))

    def __eq__(self, other: Any) -> bool:
        """
        Checks if another object is equal to this Translation instance based on the key and translation function.

        Args:
            other (Any): The object to compare with this Translation instance.

        Returns:
            bool: True if 'other' is an instance of Translation and has the same key and translation function, False otherwise.
        """
        return isinstance(other, Translation) and self.key == other.key and self.translate_func == other.translate_func

    def __repr__(self) -> str:
        """
        Provides the official string representation of the Translation object.

        Returns:
            str: The string representation of the Translation instance, e.g., "Translation(key)".
        """
        return f"Translation({self.key})"
