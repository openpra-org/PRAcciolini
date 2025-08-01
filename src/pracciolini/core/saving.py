from typing import Callable, Any


class Saver:
    """
    A class representing a saver mechanism with a unique identifier and a save function.

    Attributes:
        key (str): The unique identifier for the saver.
        save_func (Callable[[Any, Any], Any]): The function used to perform the saving.

    Methods:
        __hash__: Returns the hash based on the saver's key and function.
        __eq__: Checks equality with another Saver object based on the key and function.
        __repr__: Provides the official string representation of the Saver object.
    """

    def __init__(self, key: str, save_func: Callable[[Any, Any], Any]):
        """
        Initializes the Saver instance with a specified key and save function.

        Args:
            key (str): The unique identifier for the saver.
            save_func (Callable[[Any, Any], Any]): The function that performs the saving.
        """
        self.key = key
        self.save_func = save_func

    def __hash__(self) -> int:
        """
        Returns the hash of the Saver instance based on its key and save function.

        Returns:
            int: The hash value of the saver's key and function.
        """
        return hash((self.key, self.save_func))

    def __eq__(self, other: Any) -> bool:
        """
        Checks if another object is equal to this Saver instance based on the key and save function.

        Args:
            other (Any): The object to compare with this Saver instance.

        Returns:
            bool: True if 'other' is an instance of Saver and has the same key and save function, False otherwise.
        """
        return (
            isinstance(other, Saver) and
            self.key == other.key and
            self.save_func == other.save_func
        )

    def __repr__(self) -> str:
        """
        Provides the official string representation of the Saver object.

        Returns:
            str: The string representation of the Saver instance, e.g., "Saver(key)".
        """
        return f"Saver({self.key})"