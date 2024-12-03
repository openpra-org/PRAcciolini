from pracciolini.grammar.openpsa.xml.expression.constants import FloatExpression
from pracciolini.grammar.openpsa.xml.reference import SystemMissionTimeParameterReference
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class ParametricExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"].children.add("float")
        kwargs["info"].req_children.add("float")
        super().__init__(*args, **kwargs)

    # @classmethod
    # def validate(cls, instance: 'ParametricExpression'):
    #     super().validate(instance)
    #     if len(instance.children) == 0:
    #         raise ValueError(f"{instance.info.tag} should contain at least one element")
    #
    #     for child in instance.children:
    #         if not isinstance(child, FloatExpression):
    #             raise ValueError(f"{child.info.tag} should be a float")
    #
    #     return instance


class UniformDeviateExpression(ParametricExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="uniform-deviate", class_type=self)
        super().__init__(*args, **kwargs)

    @property
    def min(self):
        min_float: FloatExpression = self.children[0]
        return min_float.value

    @min.setter
    def min(self, value):
        self.children[0].value = value

    @min.deleter
    def min(self):
        self.children[0].value = None


    @property
    def max(self):
        max_float: FloatExpression = self.children[1]
        return max_float.value

    @max.setter
    def max(self, value):
        self.children[1].value = value

    @max.deleter
    def max(self):
        self.children[1].value = None

    @classmethod
    def validate(cls, instance: 'UniformDeviateExpression'):
        super().validate(instance)
        if len(instance.children) != 2:
            raise ValueError(f"{instance.info.tag} should contain exactly 2 fields [min, max]")

        if not (instance.max is not None and instance.min is not None):
            raise ValueError(f"{instance.info.tag} only has one of [min={instance.min}, max={instance.max}]")

        if instance.min > instance.max:
            raise ValueError(f"{instance.info.tag} has incorrect bounds [{instance.min}, {instance.max}]")

        return instance


class SystemMissionTimeDependent(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"].children.add("system-mission-time")
        kwargs["info"].req_children.add("system-mission-time")
        super().__init__(*args, **kwargs)


class GLMExpression(ParametricExpression, SystemMissionTimeDependent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="GLM", class_type=self)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'GLMExpression'):
        #super().validate(instance)
        if len(instance.children) != 4:
            raise ValueError(f"{instance.info.tag} should contain 4 fields")

        num_floats = 0
        num_system_mission_times = 0
        for child in instance.children:
            num_floats += isinstance(child, FloatExpression)
            num_system_mission_times += isinstance(child, SystemMissionTimeParameterReference)

        if num_floats != 3:
            raise ValueError(f"{instance.info.tag} should contain 3 floats")

        if num_system_mission_times != 1:
            raise ValueError(f"{instance.info.tag} should contain <system-mission-time/>")

        return instance


class BetaDeviateExpression(ParametricExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="beta-deviate", class_type=self)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'BetaDeviateExpression'):
        super().validate(instance)
        if len(instance.children) < 2 or len(instance.children) > 3:
            raise ValueError(f"{instance.info.tag} should contain between 2 and 3 fields")

        for child in instance.children:
            if not isinstance(child, FloatExpression):
                raise ValueError(f"{child.info.tag} should be a float")

        return instance


class GammaDeviateExpression(ParametricExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="gamma-deviate", class_type=self)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'GammaDeviateExpression'):
        super().validate(instance)
        if len(instance.children) < 2 or len(instance.children) > 3:
            raise ValueError(f"{instance.info.tag} should contain between 2 and 3 fields")

        for child in instance.children:
            if not isinstance(child, FloatExpression):
                raise ValueError(f"{child.info.tag} should be a float")

        return instance


class LognormalDeviateExpression(ParametricExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="lognormal-deviate", class_type=self)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'LognormalDeviateExpression'):
        super().validate(instance)
        if len(instance.children) < 2 or len(instance.children) > 3:
            raise ValueError(f"{instance.info.tag} should contain between 2 and 3 fields")

        for child in instance.children:
            if not isinstance(child, FloatExpression):
                raise ValueError(f"{child.info.tag} should be a float")

        return instance


class NormalDeviateExpression(ParametricExpression):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="normal-deviate", class_type=self)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate(cls, instance: 'NormalDeviateExpression'):
        super().validate(instance)
        if len(instance.children) < 2 or len(instance.children) > 3:
            raise ValueError(f"{instance.info.tag} should contain between 2 and 3 fields")

        for child in instance.children:
            if not isinstance(child, FloatExpression):
                raise ValueError(f"{child.info.tag} should be a float")

        return instance


class WeibullExpression(ParametricExpression, SystemMissionTimeDependent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="Weibull",
                                 class_type=self,
                                 children={"lognormal-deviate", "float"},
                                 req_children={"lognormal-deviate", "float"})
        super().__init__(*args, **kwargs)


class ExponentialExpression(SystemMissionTimeDependent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="exponential",
                                 class_type=self,
                                 children={"float", "parameter"})
        super().__init__(*args, **kwargs)


class PeriodicTestExpression(SystemMissionTimeDependent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="periodic-test", class_type=self, children={"float", "bool"}, req_children={"float"})
        super().__init__(*args, **kwargs)