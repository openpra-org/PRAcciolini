from abc import ABC, abstractmethod

# tag, attribute info, children info, constraints
class XmlElementConstraints(ABC):

    @property
    @abstractmethod
    def tags(self):
        pass

    @property
    @abstractmethod
    def text(self):
        pass

    @property
    @abstractmethod
    def attributes(self):
        pass

    @property
    @abstractmethod
    def children(self):
        pass



# class XmlRegistry:
#
#
#     @staticmethod
#     def register():