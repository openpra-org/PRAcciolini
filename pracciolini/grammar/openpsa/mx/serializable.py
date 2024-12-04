from abc import ABC, abstractmethod
from lxml.etree import Element

class XMLSerializable(ABC):
    @classmethod
    @abstractmethod
    def from_xml(cls, element: Element):
        pass

    @abstractmethod
    def to_xml(self) -> Element:
        pass