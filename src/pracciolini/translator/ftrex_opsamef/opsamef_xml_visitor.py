import warnings

from collections import Counter

from antlr4.tree.Tree import TerminalNodeImpl
from lxml import etree

from pracciolini.grammar.ftrex.ftp.parser import ftrex_ftpParser
from pracciolini.grammar.ftrex.ftp.visitor import ftrex_ftpVisitor

__all__ = ['OpsaMefXmlVisitor', ]


class RedefinitionWarning(RuntimeWarning):
    """Warning issued when a gate or basic event is defined more than once."""
    pass


class NoDefinitionWarning(RuntimeWarning):
    """Warning issued for an event that is referenced but never defined."""
    pass


class UnaryGateWarning(RuntimeWarning):
    """Warning issued when a boolean gate with a single child is removed."""
    pass


def check_referenced_are_defined(ref_counter: Counter[str], def_counter: Counter[str]) -> None:
    """Check if all referenced events have at least one definition.

    Parameters
    ----------
    def_counter : collections.Counter[str]
        A counter tracking all event definitions.
    ref_counter : collections.Counter[str]
        A counter tracking all event references.

    Warns
    -----
    NoDefinitionWarning
        If an event is found in the reference counter but has a count of zero
        in the definition counter.

    """
    for name in ref_counter:
        if ref_counter[name] > 0 >= def_counter[name]:
            warnings.warn(f"Event {name} is referenced but not defined", NoDefinitionWarning)


class OpsaMefXmlVisitor(ftrex_ftpVisitor):
    """An ANTLR visitor that transforms an ftrex_ftp parse tree into OPSA-MEF XML.

    This visitor walks the parse tree, building an `lxml` tree. It maintains
    state to track event definitions and references, allowing it to perform
    semantic checks and sanitize event names for XML compliance.

    Attributes
    ----------
    _event_def_counter : Counter[str]
        Tracks the number of times each sanitized event name is defined.
        Used to detect redefinitions.
    _event_ref_counter : Counter[str]
        Tracks the number of times each sanitized event name is referenced.
        Used to detect undefined events.
    _event_names : dict[str, str]
        A mapping from original event IDs in the source file to their unique,
        sanitized counterparts (e.g., 'E0').

    """
    _event_def_counter: Counter[str]
    _event_ref_counter: Counter[str]
    _event_names: dict[str, str]

    def __init__(self):
        """Initializes the visitor and its state-tracking attributes."""
        self._event_def_counter = Counter()
        self._event_ref_counter = Counter()
        self._event_names = {}
        super().__init__()

    def visit(self, tree: ftrex_ftpParser.File_Context) -> etree.ElementTree:
        """Begin visiting a parse tree.

        This is the main entry point. It clears all state from any previous
        runs and initiates the visit, wrapping the final XML element in an
        `ElementTree`.

        Parameters
        ----------
        tree : antlr4.tree.Tree.ParseTree
            The ANTLR parse tree to visit.

        Returns
        -------
        etree.ElementTree
            An `lxml.etree.ElementTree` representing the complete OPSA-MEF XML.

        """
        self._event_def_counter.clear()
        self._event_ref_counter.clear()
        self._event_names.clear()
        return etree.ElementTree(super().visit(tree))

    def visitFile_(self, ctx: ftrex_ftpParser.File_Context) -> etree.Element:
        """Visit the root of the file and create the top-level XML element.

        Parameters
        ----------
        ctx : ftrex_ftpParser.File_Context
            The parse tree context for the `file_` rule.

        Returns
        -------
        etree.Element
            The root `<opsa-mef>` lxml element.

        """
        opsamef = etree.Element("opsa-mef")

        for section_ctx in ctx.section():
            if (section := self.visitSection(section_ctx)) is not None:
                opsamef.append(section)

        check_referenced_are_defined(self._event_ref_counter, self._event_def_counter)

        return opsamef

    #
    # Parsing Main Boolean Formula
    #

    def visitTreeSection(self, ctx: ftrex_ftpParser.TreeSectionContext) -> etree.Element:
        """Visit the fault tree section and generate `<define-fault-tree>`.

        This method also performs an optimization step to identify and remove
        "unary gates" (e.g., an 'and' gate with only one child event). It warns
        the user and replaces all references to the removed gate with its child.

        Parameters
        ----------
        ctx : ftrex_ftpParser.TreeSectionContext
            The parse tree context for the `treeSection` rule.

        Returns
        -------
        etree.Element
            The `<define-fault-tree>` lxml element.

        Warns
        -----
        UnaryGateWarning
            When a redundant gate with a single child is found and removed.

        """
        fault_tree = etree.Element("define-fault-tree", name="default")

        gates_iter = filter(lambda g: g is not None, map(self.visitGate, ctx.gate()))

        # Filter out `and`, `or`, and `atleast` gates that have exactly one child
        gates_removed: dict[str, str] = {}
        for gate in gates_iter:
            bool_expr: etree.Element = gate[-1]
            # //element_tag[count(*) = 1 and *[1]/self::child_tag_name]
            # //*[not(self::tagName)]
            assert bool_expr.tag != "not"
            if len(bool_expr) < 2 and bool_expr[0].tag == 'event':
                removed: str = gate.attrib['name']
                replacement: str = bool_expr[0].attrib['name']
                gates_removed[removed] = replacement

                self._event_def_counter[removed] -= 1
                self._event_ref_counter[replacement] -= 1

                # label_removed: str = gate[0].text
                # warnings.warn(f"Removing gate `{removed}` ({label_removed}), "
                #               f"references replaced by `{replacement}`", UnaryGateWarning)
            else:
                fault_tree.append(gate)

        # Replace event references of removed gates
        for event_ref in fault_tree.iter('event'):
            if (removed := event_ref.attrib['name']) in gates_removed:
                self._event_ref_counter[removed] -= 1

                replacement: str = gates_removed[removed]
                while replacement in gates_removed:
                    replacement = gates_removed[replacement]

                gates_removed[removed] = replacement
                event_ref.attrib['name'] = replacement
                self._event_ref_counter[replacement] += 1

        for key in gates_removed:
            assert self._event_ref_counter[key] == 0, f'{key} : {self._event_ref_counter[key]}'
            assert self._event_def_counter[key] == 0, f'{key} : {self._event_def_counter[key]}'

        return fault_tree

    #
    # Gate Visitors
    #

    def visitGate(self, ctx: ftrex_ftpParser.GateContext) -> etree.Element | None:
        """Visit a gate definition.

        Parameters
        ----------
        ctx : ftrex_ftpParser.GateContext
            The parse tree context for the `gate` rule.

        Returns
        -------
        etree.Element or None
            A `<define-gate>` lxml element, or None if the gate is a
            redefinition and should be skipped.

        """
        match self.visitGateId(ctx.gateId()):
            case [name, label]:
                gate = etree.Element("define-gate", name=name)
                gate.append(label)

                gate_def = self.visitGateDef(ctx.gateDef())
                gate.append(gate_def)
                return gate

            case None:
                return None

        assert False, "Unreachable: unknown Gate ID"

    def visitGateId(self, ctx: ftrex_ftpParser.GateIdContext) -> tuple[str, etree.Element] | None:
        return self._visitEventDef(ctx.EVENT_ID(), event_type='Gate')

    def visitGateDef(self, ctx: ftrex_ftpParser.GateDefContext) -> etree.Element:
        """Visit the definition part of a gate (its type and operands).

         Parameters
         ----------
         ctx : ftrex_ftpParser.GateDefContext
             The parse tree context for the `gateDef` rule.

         Returns
         -------
         etree.Element
             An lxml element for the gate's logic (e.g., `<and>`, `<or>`).

        """
        gate = self.visitGateType(ctx.gateType())
        operands = self.visitOperands(ctx.operands())
        gate.extend(operands)
        return gate

    def visitGateType(self, ctx: ftrex_ftpParser.GateTypeContext) -> etree.Element:
        if ctx.AND():
            return etree.Element("and")

        if ctx.OR():
            return etree.Element("or")

        if ctx.ATLEAST():
            return etree.Element("atleast", min=f"{int(ctx.getText())}")

        assert False, f"Unreachable: unknown gate type: {ctx.getText()}"

    def visitOperands(self, ctx: ftrex_ftpParser.OperandsContext) -> list[etree.Element]:
        """Visit a list of gate operands.

        Parameters
        ----------
        ctx : ftrex_ftpParser.OperandsContext
            The parse tree context for the `operands` rule.

        Returns
        -------
        list[etree.Element]
            A list of lxml elements representing the child events/literals.

        """
        return [self.visitLiteral(literal_ctx) for literal_ctx in ctx.literal()]

    #
    # Parsing Probabilities associated with Basic Events
    #

    def visitImportSection(self, ctx: ftrex_ftpParser.ImportSectionContext) -> etree.Element:
        """Visit the import section containing basic event data.

        Parameters
        ----------
        ctx : ftrex_ftpParser.ImportSectionContext
            The parse tree context for the `importSection` rule.

        Returns
        -------
        etree.Element
            A `<model-data>` lxml element containing basic event definitions.

        """
        model_data = etree.Element("model-data")
        model_data.extend(self.visitImportCommands(ctx.importCommands()))
        return model_data

    def visitImportCommands(self, ctx: ftrex_ftpParser.ImportCommandsContext) -> list[etree.Element]:
        """Visit the list of basic event definitions within an import section.

        Parameters
        ----------
        ctx : ftrex_ftpParser.ImportCommandsContext
            The parse tree context for the `importCommands` rule.

        Returns
        -------
        list[etree.Element]
            A list of `<define-basic-event>` lxml elements.

        """
        basic_events = []
        for be_ctx in ctx.basicEvent():
            if (basic_event := self.visitBasicEvent(be_ctx)) is not None:
                basic_events.append(basic_event)
        return basic_events

    #
    # Basic Event Visitors
    #

    def visitBasicEvent(self, ctx: ftrex_ftpParser.BasicEventContext) -> etree.Element | None:
        """Visit a basic event definition.

        Parameters
        ----------
        ctx : ftrex_ftpParser.BasicEventContext
            The parse tree context for the `basicEvent` rule.

        Returns
        -------
        etree.Element or None
            A `<define-basic-event>` element, or None if it's a redefinition.

        """
        match self.visitBasicEventID(ctx.basicEventID()):
            case [name, label]:
                basic_event = etree.Element("define-basic-event", name=name)
                basic_event.append(label)

                probability = self.visitProbability(ctx.probability())
                basic_event.append(probability)
                return basic_event

            case None:
                return None

        assert False, "Unreachable: unknown basic event id"

    def visitBasicEventID(self, ctx: ftrex_ftpParser.BasicEventIDContext) -> etree.Element | None:
        return self._visitEventDef(ctx.EVENT_ID(), event_type='Basic Event')

    def visitProbability(self, ctx: ftrex_ftpParser.ProbabilityContext) -> etree.Element:
        """Visit a probability value.

        Parameters
        ----------
        ctx : ftrex_ftpParser.ProbabilityContext
            The parse tree context for the `probability` rule.

        Returns
        -------
        etree.Element
            A `<float>` lxml element with the probability as its value.

        """
        return etree.Element("float", value=ctx.REAL_NUMBER().getText())

    #
    # Event Definition Helper
    #

    def _visitEventDef(self, event_id: TerminalNodeImpl, event_type: str = 'Event', name_prefix: str | None = None) -> tuple[str, etree.Element] | None:
        """Handle the definition of any event (gate or basic).

        This helper sanitizes the event ID, checks for redefinitions, and
        updates the definition counter.

        Parameters
        ----------
        event_id : antlr4.tree.Tree.TerminalNodeImpl
            The ANTLR terminal node for the event's ID.
        event_type : str, optional
            A string ('Gate', 'Basic Event') for clear warning messages,
            by default 'Event'.
        name_prefix : str | None, default = None
            The prefix string for the sanitized event name.

        Returns
        -------
        tuple[str, etree.Element] or None
            A tuple of (sanitized_name, label_element) if the definition
            is new, otherwise None.

        Warns
        -----
        RedefinitionWarning
            If the event ID has been defined previously.

        """
        orig_id = event_id.getText()

        if name_prefix is not None:
            name = self.sanitize_event_id(orig_id, prefix=name_prefix)
        else:
            name = self.sanitize_event_id(orig_id)

        self._event_def_counter[name] += 1
        if (n := self._event_def_counter[name]) > 1:
            self._event_def_counter[name] -= 1
            warnings.warn(
                f"{event_type} `{name}` ({orig_id}) is redefined ({n}) on line {event_id.getSymbol().line}: skipping"
            )
            return None

        return name, self.create_label(orig_id)

    #
    # Event Reference Visitors
    #

    def visitNotEvent(self, ctx: ftrex_ftpParser.NotEventContext) -> etree.Element:
        """Visit a negated event reference.

        Parameters
        ----------
        ctx : ftrex_ftpParser.NotEventContext
            The parse tree context for the `notEvent` rule.

        Returns
        -------
        etree.Element
            A `<not>` lxml element wrapping an `<event>` element.

        """
        not_gate = etree.Element("not")
        not_gate.append(self.visitEvent(ctx.event()))
        return not_gate

    def visitEvent(self, ctx: ftrex_ftpParser.EventContext) -> etree.Element:
        """Visit a direct event reference.

        This sanitizes the event name and increments the reference counter.

        Parameters
        ----------
        ctx : ftrex_ftpParser.EventContext
            The parse tree context for the `event` rule.

        Returns
        -------
        etree.Element
            An `<event>` lxml element with the sanitized name.

        """
        orig_name: str = ctx.EVENT_ID().getText()
        name: str = self.sanitize_event_id(orig_name)
        self._event_ref_counter[name] += 1
        return etree.Element("event", name=name)

    #
    # Helper methods
    #

    def sanitize_event_id(self, unsafe_event_id: str, *, prefix: str = 'E') -> str:
        """Convert any event ID string into a unique, simple, sanitized ID.

        Parameters
        ----------
        unsafe_event_id : str
            The original event ID from the source file.
        prefix : str, default = 'E'
            The prefix to prepend to the event ID.

        Returns
        -------
        str
            A unique, sanitized string (e.g., 'E0').

        Notes
        -----
        This method ensures that even if source IDs are complex, duplicated,
        or not XML-compliant, the resulting internal names are simple and
        unique. It memoizes the mapping in the `_event_names` attribute.

        """
        idx = f'{prefix}{len(self._event_names)}'
        return self._event_names.setdefault(unsafe_event_id, idx)

    @staticmethod
    def create_label(text: str) -> etree.Element:
        """Create a lxml <label> element with the given text.

        Parameters
        ----------
        text : str
            The text content for the label.

        Returns
        -------
        etree.Element
            An `lxml.etree.Element` for the label.

        """
        label = etree.Element("label")
        label.text = text
        return label
