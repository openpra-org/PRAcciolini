import unittest
from typing import Any
from pracciolini.utils import FilterMixin
import unittest


class TestFilterMixin(unittest.TestCase):
    def setUp(self):
        self.filter_mixin = FilterMixin()

    def test_add_filter_with_non_callable(self):
        """
        Test adding a non-callable filter should raise ValueError.
        """
        with self.assertRaises(ValueError):
            self.filter_mixin.add_filter("not a callable")

    def test_add_filter_with_callable(self):
        """
        Test adding a callable filter should succeed without errors.
        """
        def sample_filter(x):
            return x
        self.filter_mixin.add_filter(sample_filter)
        self.assertEqual(len(self.filter_mixin.filters), 1)

    def test_apply_filters_no_filters(self):
        """
        Test applying filters when no filters have been added.
        """
        data = [1, 2, 3]
        result = list(self.filter_mixin.apply_filters(data))
        self.assertEqual(result, data)

    def test_apply_filters_single_filter(self):
        """
        Test applying a single filter that modifies the data.
        """
        def add_one(x):
            return x + 1
        self.filter_mixin.add_filter(add_one)
        data = [1, 2, 3]
        expected = [2, 3, 4]
        result = list(self.filter_mixin.apply_filters(data))
        self.assertEqual(result, expected)

    def test_apply_filters_multiple_filters(self):
        """
        Test applying multiple filters.
        """
        def add_one(x):
            return x + 1

        def multiply_by_two(x):
            return x * 2
        self.filter_mixin.add_filter(add_one)
        self.filter_mixin.add_filter(multiply_by_two)
        data = [1, 2, 3]
        expected = [4, 6, 8]  # (1+1)*2, (2+1)*2, (3+1)*2
        result = list(self.filter_mixin.apply_filters(data))
        self.assertEqual(result, expected)

    def test_apply_filters_exclude_items(self):
        """
        Test applying filters that exclude items (return None).
        """
        def exclude_odds(x):
            if x % 2 != 0:
                return None
            return x
        self.filter_mixin.add_filter(exclude_odds)
        data = [1, 2, 3, 4, 5]
        expected = [2, 4]
        result = list(self.filter_mixin.apply_filters(data))
        self.assertEqual(result, expected)

    def test_apply_filters_modify_and_exclude(self):
        """
        Test filters that modify and exclude items.
        """
        def add_one(x):
            return x + 1

        def exclude_odds(x):
            if x % 2 != 0:
                return None
            return x

        self.filter_mixin.add_filter(add_one)
        self.filter_mixin.add_filter(exclude_odds)
        data = [1, 2, 3, 4]
        expected = [2, 4]  # 1+1=2 (included), 2+1=3 (excluded), 3+1=4 (included), 4+1=5 (excluded)
        result = list(self.filter_mixin.apply_filters(data))
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
