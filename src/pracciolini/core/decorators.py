import functools
import inspect
from typing import Any, Callable, TypeVar, cast

from pracciolini.core.grammar import Grammar
from pracciolini.core.translation import Translation
from pracciolini.core.registry import GrammarRegistry
from pracciolini.core.validation import Validation
from pracciolini.core.loading import Loader
from pracciolini.core.saving import Saver

T = TypeVar('T')


def translation(source: str, target: str) -> Callable[[Callable[[T], T]], Callable[[T], T]]:
    """
    A decorator for defining translations between grammars. This decorator registers the decorated function
    as a translation function in the GrammarRegistry and enforces that the function has the correct signature.

    Args:
        source (str): The key of the source grammar.
        target (str): The key of the target grammar.

    Returns:
        Callable[[Callable[[T], T]], Callable[[T], T]]: A decorator that takes a translation function and returns
        a wrapped version of that function.

    Raises:
        TypeError: If the decorated function does not accept exactly one argument or does not specify a return type.
    """

    def decorator(func: Callable[[T], T]) -> Callable[[T], T]:
        """
        The actual decorator that wraps the translation function and registers it.

        Args:
            func (Callable[[T], T]): The function to be decorated. It must accept exactly one argument of any type
            and return a value of the same type.

        Returns:
            Callable[[T], T]: The wrapped function.

        Raises:
            TypeError: If the function signature does not meet the requirements.
        """
        # Check function signature to enforce argument and return types
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1 or sig.return_annotation is sig.empty:
            raise TypeError("translation must accept exactly one argument and must specify a return type")

        # Generate a unique translation key based on grammar keys and function name
        translation_key = f"{func.__name__}({source}) -> {target}"

        # Add the translation to the registry
        GrammarRegistry.add_translation(Grammar(source), Grammar(target), Translation(translation_key, func))

        @functools.wraps(func)
        def wrapper(*args: T, **kwargs: Any) -> T:
            """
            The wrapper function that calls the original function.

            Args:
                *args (T): The arguments to pass to the function.
                **kwargs (Any): The keyword arguments to pass to the function.

            Returns:
                T: The result of the function call.
            """
            return func(*args, **kwargs)

        return cast(Callable[[T], T], wrapper)

    return decorator

def validation(grammar_key: str) -> Callable[[Callable[[T], Any]], Callable[[T], Any]]:
    """
    A decorator for defining validations for a grammar. This decorator registers the decorated function
    as a validation function in the GrammarRegistry and enforces that the function has the correct signature.

    Args:
        grammar_key (str): The key of the grammar for which this validation is defined.

    Returns:
        Callable[[Callable[[T], Any]], Callable[[T], Any]]: A decorator that takes a validation function and returns
        a wrapped version of that function.

    Raises:
        TypeError: If the decorated function does not accept exactly one argument or does not specify a return type.
    """

    def decorator(func: Callable[[T], Any]) -> Callable[[T], Any]:
        """
        The actual decorator that wraps the validation function and registers it.

        Args:
            func (Callable[[T], Any]): The function to be decorated. It must accept exactly one argument of any type
            and specify a return type.

        Returns:
            Callable[[T], Any]: The wrapped function.

        Raises:
            TypeError: If the function signature does not meet the requirements.
        """
        # Check function signature to enforce argument and return types
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1 or sig.return_annotation is sig.empty:
            raise TypeError("validation must accept exactly one argument and must specify a return type")

        # Generate a unique validation key based on grammar key and function name
        validation_key = f"{func.__name__}({grammar_key})"

        # Add the validation to the registry
        GrammarRegistry.add_validation(Grammar(grammar_key), Validation(validation_key, func))

        @functools.wraps(func)
        def wrapper(arg: T) -> Any:
            """
            The wrapper function that calls the original function.

            Args:
                arg (T): The argument to pass to the function.

            Returns:
                Any: The result of the function call.
            """
            return func(arg)

        return wrapper

    return decorator

## TODO:: Add validation decorator. Think about whether there should be one validation method per Grammar, or a set,
## and whether the validation is set to pass iff all the validation rules pass.

def load(grammar_key: str, extension: str) -> Callable[[Callable[[Any], T]], Callable[[Any], T]]:
    """
    A decorator for defining loaders for a grammar. This decorator registers the decorated function
    as a loader function in the GrammarRegistry and enforces that the function has the correct signature.

    Args:
        grammar_key (str): The key of the grammar for which this loader is defined.
        extension (str): The file extension that this loader can handle.

    Returns:
        Callable[[Callable[[Any], T]], Callable[[Any], T]]: A decorator that takes a loader function and returns
        a wrapped version of that function.

    Raises:
        TypeError: If the decorated function does not accept exactly one argument or does not specify a return type.
    """

    def decorator(func: Callable[[Any], T]) -> Callable[[Any], T]:
        """
        The actual decorator that wraps the loader function and registers it.

        Args:
            func (Callable[[Any], T]): The function to be decorated. It must accept exactly one argument
            (e.g., a file path or file-like object) and return an object of type T.

        Returns:
            Callable[[Any], T]: The wrapped function.

        Raises:
            TypeError: If the function signature does not meet the requirements.
        """
        # Check function signature to enforce argument and return types
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1 or sig.return_annotation is sig.empty:
            raise TypeError("Loader must accept exactly one argument and must specify a return type")

        # Generate a unique loader key based on grammar key, extension, and function name
        loader_key = f"{func.__name__}({grammar_key}, .{extension})"

        # Add the loader to the registry
        GrammarRegistry.add_loader(Grammar(grammar_key), extension, Loader(loader_key, func))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """
            The wrapper function that calls the original loader function.

            Args:
                *args (Any): The arguments to pass to the function.
                **kwargs (Any): The keyword arguments to pass to the function.

            Returns:
                T: The result of the function call.
            """
            return func(*args, **kwargs)

        return cast(Callable[[Any], T], wrapper)

    return decorator


def save(grammar_key: str, extension: str) -> Callable[[Callable[[T, Any], Any]], Callable[[T, Any], Any]]:
    """
    A decorator for defining savers for a grammar. This decorator registers the decorated function
    as a saver function in the GrammarRegistry and enforces that the function has the correct signature.

    Args:
        grammar_key (str): The key of the grammar for which this saver is defined.
        extension (str): The file extension that this saver can handle.

    Returns:
        Callable[[Callable[[T, Any], Any]], Callable[[T, Any], Any]]: A decorator that takes a saver function and returns
        a wrapped version of that function.

    Raises:
        TypeError: If the decorated function does not accept exactly two arguments or does not specify a return type.
    """

    def decorator(func: Callable[[T, Any], Any]) -> Callable[[T, Any], Any]:
        """
        The actual decorator that wraps the saver function and registers it.

        Args:
            func (Callable[[T, Any], Any]): The function to be decorated. It must accept exactly two arguments: the object
            to save and the destination (e.g., file path or file-like object), and must specify a return type.

        Returns:
            Callable[[T, Any], Any]: The wrapped function.

        Raises:
            TypeError: If the function signature does not meet the requirements.
        """
        # Check function signature to enforce argument and return types
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 2 or sig.return_annotation is sig.empty:
            raise TypeError("Saver must accept exactly two arguments and must specify a return type")

        # Generate a unique saver key based on grammar key, extension, and function name
        saver_key = f"{func.__name__}({grammar_key}, .{extension})"

        # Add the saver to the registry
        GrammarRegistry.add_saver(Grammar(grammar_key), extension, Saver(saver_key, func))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            The wrapper function that calls the original saver function.

            Args:
                *args (Any): The arguments to pass to the function.
                **kwargs (Any): The keyword arguments to pass to the function.

            Returns:
                Any: The result of the function call.
            """
            return func(*args, **kwargs)

        return wrapper

    return decorator