import warnings

from collections import Counter

from lxml import etree

from pracciolini.grammar.ftrex.ftp.parser import ftrex_ftpParser
from pracciolini.grammar.ftrex.ftp.visitor import ftrex_ftpVisitor

class RedefinitionWarning(RuntimeWarning):
    pass


class NoDefinitionWarning(RuntimeWarning):
    pass


class OpsaMefXmlVisitor(ftrex_ftpVisitor):
    _event_def_counter: Counter[str]
    _event_ref_counter: Counter[str]

    def __init__(self):
        self._event_def_counter = Counter()
        self._event_ref_counter = Counter()
        super().__init__()

    def visit(self, tree):
        self._event_def_counter.clear()
        self._event_ref_counter.clear()
        return super().visit(tree)

    @staticmethod
    def sanitize_event_id(unsafe_event_id: str) -> str:
        # return element, gate_type_element
        return unsafe_event_id

    def visitFile_(self, ctx:ftrex_ftpParser.File_Context):
        opsamef = etree.Element("opsa-mef")
        for section_ctx in ctx.section():
            if (section := self.visitSection(section_ctx)) is not None:
                opsamef.append(section)

        references = (
            name for name in self._event_ref_counter.keys()
            if self._event_ref_counter[name] > 0
        )

        for name in references:
            if self._event_def_counter[name] < 1:
                warnings.warn(f"Event {name} is referenced but not defined", NoDefinitionWarning)
        return opsamef

    def visitTreeSection(self, ctx: ftrex_ftpParser.TreeSectionContext):
        fault_tree = etree.Element("define-fault-tree", name="default")
        for gate_ctx in ctx.gate():
            if (gate := self.visitGate(gate_ctx)) is not None:
                fault_tree.append(gate)
        return fault_tree

    def visitImportSection(self, ctx:ftrex_ftpParser.ImportSectionContext):
        model_data = etree.Element("model-data")
        model_data.extend(self.visitImportCommands(ctx.importCommands()))
        return model_data

    def visitImportCommands(self, ctx:ftrex_ftpParser.ImportCommandsContext):
        basic_events = []
        for be_ctx in ctx.basicEvent():
            if (basic_event := self.visitBasicEvent(be_ctx)) is not None:
                basic_events.append(basic_event)
        return basic_events

    def visitBasicEvent(self, ctx:ftrex_ftpParser.BasicEventContext):
        name = self.visitBasicEventID(ctx.basicEventID())

        self._event_def_counter[name] += 1
        if (n := self._event_def_counter[name]) > 1:
            warnings.warn(f"Basic Event {name} is redefined ({n}) on line {ctx.start.line}: skipping", RedefinitionWarning)
            return None

        basic_event = etree.Element("define-basic-event", name=name)

        probability = self.visitProbability(ctx.probability())
        basic_event.append(probability)

        return basic_event

    def visitBasicEventID(self, ctx:ftrex_ftpParser.BasicEventIDContext):
        return self.sanitize_event_id(ctx.EVENT_ID().getText())

    def visitProbability(self, ctx:ftrex_ftpParser.ProbabilityContext):
        return etree.Element("float", value=ctx.REAL_NUMBER().getText())

    def visitGate(self, ctx: ftrex_ftpParser.GateContext):
        name = self.visitGateId(ctx.gateId())

        self._event_def_counter[name] += 1
        if (n := self._event_def_counter[name]) > 1:
            warnings.warn(f"Gate {name} is redefined ({n}) on line {ctx.start.line}: skipping")
            return None

        gate_def = self.visitGateDef(ctx.gateDef())
        gate = etree.Element("define-gate", name=name)
        gate.append(gate_def)

        return gate

    def visitGateId(self, ctx:ftrex_ftpParser.GateIdContext):
        return self.sanitize_event_id(ctx.EVENT_ID().getText())

    def visitGateDef(self, ctx: ftrex_ftpParser.GateDefContext):
        gate = self.visitGateType(ctx.gateType())
        operands = self.visitOperands(ctx.operands())
        gate.extend(operands)
        return gate

    def visitGateType(self, ctx: ftrex_ftpParser.GateTypeContext):
        if ctx.AND():
            return etree.Element("and")

        if ctx.OR():
            return etree.Element("or")

        if ctx.ATLEAST():
            return etree.Element("atleast", min=f"{int(ctx.getText())}")

        raise ValueError

    def visitOperands(self, ctx: ftrex_ftpParser.OperandsContext):
        return [
            self.visitLiteral(literal_ctx)
            for literal_ctx in ctx.literal()
        ]

    def visitEvent(self, ctx: ftrex_ftpParser.EventContext):
        name = self.sanitize_event_id(ctx.EVENT_ID().getText())
        self._event_ref_counter[name] += 1
        return etree.Element("event", name=name)

    def visitNotEvent(self, ctx: ftrex_ftpParser.NotEventContext):
        not_gate = etree.Element("not")
        not_gate.append(self.visitEvent(ctx.event()))
        return not_gate
