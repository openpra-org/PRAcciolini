import lxml
from lxml import etree


class GateReference:
    def __init__(self, dot_path: str):
        self.dot_path = dot_path

    def to_xml(self):
        element = etree.Element("gate")
        element.set("name", self.dot_path)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'GateReference':
        if root is None:
            raise lxml.etree.ParserError(root)
        dot_path = root.get("name")
        if dot_path is None:
            raise lxml.etree.ParserError("attr name is missing in gate ref")
        gate_ref: GateReference = GateReference(dot_path=dot_path)
        return gate_ref

    def __str__(self):
        return f"gate-ref.dot_path: {self.dot_path}"

class BasicEventReference:
    def __init__(self, dot_path: str):
        self.dot_path = dot_path

    def to_xml(self):
        element = etree.Element("basic-event")
        element.set("name", self.dot_path)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'BasicEventReference':
        if root is None:
            raise lxml.etree.ParserError(root)
        dot_path = root.get("name")
        if dot_path is None:
            raise lxml.etree.ParserError("attr name is missing in basic-event ref")
        basic_event_ref: BasicEventReference = BasicEventReference(dot_path=dot_path)
        return basic_event_ref

    def __str__(self):
        return f"basic-event-ref.dot_path: {self.dot_path}"