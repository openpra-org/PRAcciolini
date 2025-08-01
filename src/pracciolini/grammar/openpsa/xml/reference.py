from pracciolini.grammar.openpsa.xml.define_event import NamedEvent
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class ParameterReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="parameter", class_type=self, attrs={'unit'})
        super().__init__(*args, **kwargs)


class SystemMissionTimeParameterReference(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="system-mission-time", class_type=self)
        super().__init__(*args, **kwargs)


class ExternalFunctionReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="extern-function", class_type=self, children={"float", "extern-function"})
        super().__init__(*args, **kwargs)


class BasicEventReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="basic-event",
                                 class_type=self)
        super().__init__(*args, **kwargs)


class HouseEventReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="house-event",
                                 class_type=self)
        super().__init__(*args, **kwargs)


class GenericEventReference(NamedEvent):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="event",
                                 attrs={"type"},
                                 class_type=self)
        super().__init__(*args, **kwargs)
