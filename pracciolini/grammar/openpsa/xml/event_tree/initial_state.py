import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.event_tree.fork import ForkDefinition
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class InitialStateDefinition(XMLSerializable):
    def __init__(self, fork: ForkDefinition):
        self.fork = fork  # ForkDefinition instance

    def to_xml(self):
        element = etree.Element("initial-state")
        element.append(self.fork.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'InitialStateDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)
        fork_xml = root.find("fork")
        if fork_xml is None:
            raise lxml.etree.ParserError("initial-state does not contain a fork event")
        return cls(fork=ForkDefinition.from_xml(fork_xml))

    def __str__(self):
        return (f"\ninitial-state:"
                f"\n\t{self.fork}")