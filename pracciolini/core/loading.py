from typing import Callable, Any


class Loader:
    """
    A class representing a loader mechanism with a unique identifier and a load function.

    Attributes:
        key (str): The unique identifier for the loader.
        load_func (Callable[[Any], Any]): The function used to perform the loading.

    Methods:
        __hash__: Returns the hash based on the loader's key and function.
        __eq__: Checks equality with another Loader object based on the key and function.
        __repr__: Provides the official string representation of the Loader object.
    """

    def __init__(self, key: str, load_func: Callable[[Any], Any]):
        """
        Initializes the Loader instance with a specified key and load function.

        Args:
            key (str): The unique identifier for the loader.
            load_func (Callable[[Any], Any]): The function that performs the loading.
        """
        self.key = key
        self.load_func = load_func

    def __hash__(self) -> int:
        """
        Returns the hash of the Loader instance based on its key and load function.

        Returns:
            int: The hash value of the loader's key and function.
        """
        return hash((self.key, self.load_func))

    def __eq__(self, other: Any) -> bool:
        """
        Checks if another object is equal to this Loader instance based on the key and load function.

        Args:
            other (Any): The object to compare with this Loader instance.

        Returns:
            bool: True if 'other' is an instance of Loader and has the same key and load function, False otherwise.
        """
        return (
            isinstance(other, Loader) and
            self.key == other.key and
            self.load_func == other.load_func
        )

    def __repr__(self) -> str:
        """
        Provides the official string representation of the Loader object.

        Returns:
            str: The string representation of the Loader instance, e.g., "Loader(key)".
        """
        return f"Loader({self.key})"
