from typing import Callable, Any


class Validation:
    """
    A class representing a validation mechanism with a unique identifier and a validation function.

    Attributes:
        key (str): The unique identifier for the validation.
        validate_func (Callable[[Any], Any]): The function used to perform the validation.

    Methods:
        __hash__: Returns the hash based on the validation's key and function.
        __eq__: Checks equality with another Validation object based on the key and function.
        __repr__: Provides the official string representation of the Validation object.
    """

    def __init__(self, key: str, validate_func: Callable[[Any], Any]):
        """
        Initializes the Validation instance with a specified key and validation function.

        Args:
            key (str): The unique identifier for the validation.
            validate_func (Callable[[Any], Any]): The function that performs the validation.
        """
        self.key = key
        self.validate_func = validate_func

    def __hash__(self) -> int:
        """
        Returns the hash of the Validation instance based on its key and validation function.

        Returns:
            int: The hash value of the validation's key and function.
        """
        return hash((self.key, self.validate_func))

    def __eq__(self, other: Any) -> bool:
        """
        Checks if another object is equal to this Validation instance based on the key and validation function.

        Args:
            other (Any): The object to compare with this Validation instance.

        Returns:
            bool: True if 'other' is an instance of Validation and has the same key and validation function, False otherwise.
        """
        return (
            isinstance(other, Validation) and
            self.key == other.key and
            self.validate_func == other.validate_func
        )

    def __repr__(self) -> str:
        """
        Provides the official string representation of the Validation object.

        Returns:
            str: The string representation of the Validation instance, e.g., "Validation(key)".
        """
        return f"Validation({self.key})"
