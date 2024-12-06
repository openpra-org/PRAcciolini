from typing import Literal

from pracciolini.grammar.openpsa.xml.expression.constants import FloatExpression, ConstantExpression
from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class EventDefinition(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"].attrs.update({"name", "role"})
        kwargs["info"].req_attrs.add("name")
        kwargs["info"].children.update({"label", "attributes"})
        super().__init__(*args, **kwargs)

    @property
    def role(self) -> Literal["public", "private", "", None]:
        return self["role"]

    @role.setter
    def role(self, value: Literal["public", "private", "", None]):
        self["role"] = value

    @role.deleter
    def role(self):
        self["role"] = None

    @property
    def name(self) -> str:
        return self["name"]

    @name.setter
    def name(self, value: str):
        self["name"] = value

    @name.deleter
    def name(self):
        pass

    def __str__(self) -> str:
        return super().__str__()

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class NamedEvent(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"].attrs.add("name")
        kwargs["info"].req_attrs.add("name")
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return self["name"]

    @name.setter
    def name(self, name_to_set: str):
        self["name"] = name_to_set

    @name.deleter
    def name(self):
        self["name"] = ""

    @classmethod
    def validate(cls, instance: 'NamedEvent'):
        super().validate(instance)
        if instance.name is None or instance.name == "":
            raise ValueError(f"{instance.info.tag} should contain a non-empty name string")
        return instance

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)



class BasicEventDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-basic-event",
                                 class_type=self,
                                 children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'BasicEventDefinition'):
        super().validate(instance)
        return instance

    @property
    def value(self):
        for child in self.children:
            if isinstance(child, FloatExpression):
                return child.value
        return None

    @value.setter
    def value(self, value):
        for child in self.children:
            if isinstance(child, FloatExpression):
                child.value = value

    @value.deleter
    def value(self):
        for child in self.children:
            if isinstance(child, FloatExpression):
                child.value = None


class HouseEventDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-house-event",
                                 class_type=self,
                                 children={"constant"})
        super().__init__(*args, **kwargs)

    @property
    def value(self):
        for child in self.children:
            if isinstance(child, ConstantExpression):
                return child.value
        return None

    @value.setter
    def value(self, value):
        for child in self.children:
            if isinstance(child, ConstantExpression):
                child.value = value

    @value.deleter
    def value(self):
        for child in self.children:
            if isinstance(child, ConstantExpression):
                child.value = None


class ParameterDefinition(EventDefinition):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="define-parameter",
                                 class_type=self,
                                 attrs={'unit'},
                                 children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)
