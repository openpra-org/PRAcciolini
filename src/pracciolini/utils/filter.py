from typing import Callable, Generator, List, TypeVar, Optional

T = TypeVar('T')


class FilterMixin:
    """
    A mixin class that provides filtering functionality to subclasses.
    It allows adding callable filters which can modify or exclude items from data sequences.

    Attributes:
        filters (List[Callable[[T], Optional[T]]]): A list of filter functions.
    """

    def __init__(self) -> None:
        """
        Initializes the FilterMixin with an empty list of filters.
        """
        self.filters: List[Callable[[T], Optional[T]]] = []

    def add_filter(self, filter_func: Callable[[T], Optional[T]]) -> None:
        """
        Adds a filter to the list of filters.

        Parameters:
            filter_func (Callable[[T], Optional[T]]): A callable that takes an item of type T and returns a modified
            item of type T or None to exclude the item.

        Raises:
            ValueError: If the provided filter_func is not callable.

        Returns:
            None
        """
        if not callable(filter_func):
            raise ValueError("Filter must be callable")
        self.filters.append(filter_func)

    def apply_filters(self, data: List[T]) -> Generator[T, None, None]:
        """
        Applies all added filters to the input data.

        Parameters:
            data (List[T]): A list of items to which the filters will be applied.

        Yields:
            T: Filtered items that are not excluded by any filter.

        Description:
            Each item in the input data is processed through all filters in sequence.
            If a filter returns None, the item is excluded and the loop breaks for that item.
            Otherwise, the item, possibly modified, is yielded.
        """
        for item in data:
            for filter_func in self.filters:
                item = filter_func(item)
                if item is None:
                    break
            if item is not None:
                yield item
