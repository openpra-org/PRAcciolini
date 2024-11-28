from typing import Set, Optional

import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.event_tree.collect_formula import CollectFormulaDefinition
from pracciolini.grammar.openpsa.xml.event_tree.sequence import SequenceReference
from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class ForkDefinition(XMLSerializable):
    def __init__(self,
                 functional_event_ref: str,
                 paths: Optional[Set['PathDefinition']] = None
        ):
        self.functional_event_ref: str = functional_event_ref
        # maybe this is should be a dict mapping state names to paths
        self.paths: Set[PathDefinition] = paths if paths is not None else set()

    def to_xml(self):
        element = etree.Element("fork")
        element.set("functional-event", self.functional_event_ref)
        for path in self.paths:
            element.append(path.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.ElementTree) -> 'ForkDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)

        # parse path definition list
        paths_xml = root.findall("path")
        if paths_xml is None or len(paths_xml) == 0:
            raise lxml.etree.ParserError("no path definitions in fork")
        paths: Set[PathDefinition] = set()
        for path_xml in paths_xml:
            path: PathDefinition = PathDefinition.from_xml(path_xml)
            paths.add(path)

        # parse functional-event reference
        functional_event_ref = root.get("functional-event")
        if functional_event_ref is None:
            raise lxml.etree.ParserError("functional-event reference is missing in fork")

        return cls(functional_event_ref=functional_event_ref, paths=paths)

    def __str__(self):
        str_rep = [
            "fork-definition",
            f"\tfunctional-event-ref: {self.functional_event_ref}",
            f"\tnum_paths: {len(self.paths)}",
        ]
        if len(self.paths) > 0:
            paths_str = [
                "\tpaths:",
            ]
            for path in self.paths:
                paths_str.append(str(path))
            str_rep.append("\n\t".join(paths_str))
        return "\n".join(str_rep)



class PathDefinition(XMLSerializable):
    def __init__(self,
                 state: str,
                 collect_formula: CollectFormulaDefinition,
                 fork: Optional[ForkDefinition] = None,
                 sequence_ref: Optional[SequenceReference] = None
        ):
        self.state: str = state  # 'Success' or 'Failure'
        self.collect_formula: CollectFormulaDefinition = collect_formula
        self.fork: Optional[ForkDefinition] = fork
        self.sequence_ref: Optional[SequenceReference] = sequence_ref

    def to_xml(self):
        element = etree.Element("path")
        element.set("state", self.state)
        if self.collect_formula:
            element.append(self.collect_formula.to_xml())
        if self.fork:
            element.append(self.fork.to_xml())
        if self.sequence_ref:
            element.append(self.sequence_ref.to_xml())
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'PathDefinition':
        if root is None:
            raise lxml.etree.ParserError(root)

        # parse the state name for this path
        state = root.get("state")
        if state is None:
            raise lxml.etree.ParserError("no state definition in path")

        # parse collect-formula definition
        collect_formula_xml = root.find("collect-formula")
        if collect_formula_xml is None:
            raise lxml.etree.ParserError("no collect-formula in path definition")
        collect_formula: CollectFormulaDefinition = CollectFormulaDefinition.from_xml(collect_formula_xml)

        # parse fork definition
        fork_xml = root.find("fork")
        fork: Optional[ForkDefinition] = None
        if fork_xml is not None:
            fork = ForkDefinition.from_xml(fork_xml)

        # parse sequence reference
        sequence_ref_xml = root.find("sequence")
        sequence_ref: Optional[SequenceReference] = None
        if sequence_ref_xml is not None:
            sequence_ref: SequenceReference = SequenceReference.from_xml(sequence_ref_xml)

        if sequence_ref_xml is None and fork_xml is None:
            raise lxml.etree.ParserError("path definition does not specify a target fork or end-state")

        return cls(state=state, collect_formula=collect_formula, fork=fork, sequence_ref=sequence_ref)

    def __str__(self):
        str_rep = [
            "path-definition",
            f"\tstate: {self.state}",
        ]
        if self.collect_formula is not None:
            str_rep.append(f"\tcollect-formula: {self.collect_formula}")
        if self.fork is not None:
            str_rep.append(f"\tfork.functional-event-ref: {self.fork.functional_event_ref}")
        if self.sequence_ref is not None:
            str_rep.append(f"\tsequence-ref: {self.sequence_ref}")
        return "\n".join(str_rep)