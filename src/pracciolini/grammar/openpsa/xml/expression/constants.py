from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class Unary(XMLSerializable):

    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"].attrs.add("value")
        super().__init__(*args, **kwargs)

    @property
    def value(self):
        return self["value"]

    @value.setter
    def value(self, value):
        self["value"] = value

    @value.deleter
    def value(self):
        self["value"] = None


class FloatExpression(Unary):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="float", class_type=self)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'FloatExpression'):
        super().validate(instance)
        return instance


class IntegerExpression(Unary):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="int", class_type=self)
        super().__init__(*args, **kwargs)


class BoolExpression(Unary):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="bool", class_type=self)
        super().__init__(*args, **kwargs)


class ConstantExpression(Unary):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="constant", class_type=self)
        super().__init__(*args, **kwargs)


class ConstantPi(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="pi", class_type=self)
        super().__init__(*args, **kwargs)
