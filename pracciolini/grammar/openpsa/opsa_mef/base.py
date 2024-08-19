from abc import ABC, abstractmethod
from lxml import etree


class ValidatableMixin:
    @staticmethod
    def validate(value) -> bool:
        return True


class ValidatableElement(etree.ElementBase, ValidatableMixin):
    """
    An abstract base class that all custom XML elements must inherit from.
    This class requires that all subclasses implement the validate method.
    """
    @abstractmethod
    def validate(self):
        """
        Validate the current element's data.
        This method must be implemented by all subclasses.
        """
        pass
