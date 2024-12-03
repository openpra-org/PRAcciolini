from pracciolini.grammar.openpsa.xml.define_event import NamedEvent
from pracciolini.grammar.openpsa.xml.expression.constants import Unary
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class Attribute(NamedEvent, Unary):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="attribute", class_type=self, attrs={"type"})
        super().__init__(*args, **kwargs)


class Attributes(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="attributes", class_type=self, children={'attribute'}, req_children={'attribute'})
        super().__init__(*args, **kwargs)

