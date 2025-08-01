from typing import Any


class Grammar:
    """
    A class representing a grammar with a unique identifier.

    Attributes:
        key (str): The unique identifier for the grammar.

    Methods:
        __hash__: Returns the hash based on the grammar's key.
        __eq__: Checks equality with another Grammar object based on the key.
        __repr__: Provides the official string representation of the Grammar object.
    """

    def __init__(self, key: str):
        """
        Initializes the Grammar instance with a specified key.

        Args:
            key (str): The unique identifier for the grammar.
        """
        self.key = key

    def __hash__(self) -> int:
        """
        Returns the hash of the Grammar instance based on its key.

        Returns:
            int: The hash value of the grammar's key.
        """
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        """
        Checks if another object is equal to this Grammar instance based on the key.

        Args:
            other (Any): The object to compare with this Grammar instance.

        Returns:
            bool: True if 'other' is an instance of Grammar and has the same key, False otherwise.
        """
        return isinstance(other, Grammar) and self.key == other.key

    def __repr__(self) -> str:
        """
        Provides the official string representation of the Grammar object.

        Returns:
            str: The string representation of the Grammar instance, e.g., "Grammar(key)".
        """
        return f"Grammar({self.key})"
