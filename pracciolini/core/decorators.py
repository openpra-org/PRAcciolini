import functools
import inspect
from typing import Any, Callable, TypeVar, cast

from pracciolini.core.grammar import Grammar
from pracciolini.core.translation import Translation
from pracciolini.core.registry import GrammarRegistry

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

## TODO:: Add validation decorator. Think about whether there should be one validation method per Grammar, or a set,
## and whether the validation is set to pass iff all the validation rules pass.
