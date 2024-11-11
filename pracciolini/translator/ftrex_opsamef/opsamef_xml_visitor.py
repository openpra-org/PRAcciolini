from lxml import etree

from pracciolini.grammar.ftrex.ftp.parser import ftrex_ftpParser
from pracciolini.grammar.ftrex.ftp.visitor import ftrex_ftpVisitor


class OpsaMefXmlVisitor(ftrex_ftpVisitor):
    def __init__(self):
        self.opsa_mef: etree.Element = etree.Element("opsa-mef")
        self.fault_tree: etree.SubElement = etree.SubElement(self.opsa_mef, "define-fault-tree", name="default")
        self.model_data: etree.SubElement = etree.SubElement(self.opsa_mef, "model-data")

        self.define_gates = dict()
        self.define_basic_events = dict()

        super().__init__()

    def build_xml(self, tree) -> etree.ElementTree:
        parsed = self.visit(tree)
        xml_doc: etree.ElementTree = etree.ElementTree(self.opsa_mef)
        return xml_doc

    @staticmethod
    def create_define_basic_event(name, label = None, probability = None):
        basic_event = etree.Element("define-basic-event", name=name)
        if probability is not None:
            probability_elem = etree.SubElement(basic_event, "float", value=str(probability))
        if label is not None:
            label_elem = etree.SubElement(basic_event, "label")
            label_elem.text = str(label)
        return basic_event

    @staticmethod
    def create_generic_event(name):
        basic_event = etree.Element("event", name=name)
        return basic_event

    @staticmethod
    def create_gate_type(gate_type: str) -> etree.SubElement:
        if gate_type == "*":
            return etree.Element("and")
        if gate_type == "+":
            return etree.Element("or")
        return etree.Element("atleast", min=gate_type)

    @staticmethod
    def create_define_gate(name, label = None, gate_type: str = None):
        element = etree.Element("define-gate", name=name)
        gate_type_element = None
        if label is not None:
            label_elem = etree.SubElement(element, "label")
            label_elem.text = str(label)
        if gate_type is not None:
            gate_type_element = OpsaMefXmlVisitor.create_gate_type(gate_type)
            element.append(gate_type_element)
        return element, gate_type_element

    @staticmethod
    def create_schema_safe_name(unsafe_event_id: str):

        return element, gate_type_element



    def visitFile_(self, ctx: ftrex_ftpParser.File_Context):
        #print("visitFile_", ctx.getText())
        return super().visitFile_(ctx)

    def visitSection(self, ctx: ftrex_ftpParser.SectionContext):
        #print("visitSection", ctx.getText())
        return super().visitSection(ctx)

    def visitTreeSection(self, ctx: ftrex_ftpParser.TreeSectionContext):
        #print("visitTreeSection", ctx.getText(), ctx.getChildCount(), ctx.getChildren())
        for gate in ctx.gate():
            event_id = gate.gateId().getText()
            if self.fault_tree.xpath(f"./define-gate[@name='{event_id}']"):
                print(f"Warning: A gate with the name '{event_id}' already exists. Skipping.")
                continue

            gate_type = gate.gateType().getText()
            gate_element, gate_type_element = self.create_define_gate(name=event_id, gate_type=gate_type)

            # for this pass, assume all children are generic-events.
            gate_children = gate.childRefList().childRef()
            for child in gate_children:
                child_event_id = child.getText()
                child_event = self.create_generic_event(child_event_id)
                gate_type_element.append(child_event)
                # try:
                #     if gate_element.xpath(f"./event[@name='{child_event_id}']"):
                #         print(f"Warning: event'{child_event_id}' already exists on '{event_id}'. Skipping.")
                #         continue
                # except:
                #     child_event = self.create_generic_event(child_event_id)
                #     gate_element.append(child_event)

            self.fault_tree.append(gate_element)

            #print (event_id, gate_type, gate_children, gate_element)

        return super().visitTreeSection(ctx)

    def visitGate(self, ctx: ftrex_ftpParser.GateContext):
        return super().visitGate(ctx)

    def visitProcessSection(self, ctx: ftrex_ftpParser.ProcessSectionContext):
        #print("visitProcessSection", ctx.getText())
        return super().visitProcessSection(ctx)

    def visitImportSection(self, ctx: ftrex_ftpParser.ImportSectionContext):
        return super().visitImportSection(ctx)

    def visitLimitSection(self, ctx: ftrex_ftpParser.LimitSectionContext):
        return super().visitLimitSection(ctx)

    def visitProcessCommands(self, ctx: ftrex_ftpParser.ProcessCommandsContext):
        # print("visitProcessCommands", ctx.getText(), ctx.getChildCount(), ctx.getChildren())
        # for i in range(0, ctx.getChildCount()-1):
        #     print("command", ctx.EVENT_ID(i))
        return super().visitProcessCommands(ctx)

    def visitImportCommands(self, ctx: ftrex_ftpParser.ImportCommandsContext):
        # Assuming ctx.EVENT_ID() and ctx.NUMBER() return iterables of all EVENT_IDs and NUMBERs
        for event_id_ctx, number_ctx in zip(ctx.EVENT_ID(), ctx.NUMBER()):
            event_id = str(event_id_ctx.getText())
            number = str(number_ctx.getText())

            if self.model_data.xpath(f"./define-basic-event[@name='{event_id}']"):
                print(f"Warning: A basic event with the name '{event_id}' already exists. Skipping.")
                continue

            basic_event = self.create_define_basic_event(name=event_id, probability=number)
            self.model_data.append(basic_event)

        return super().visitImportCommands(ctx)

    def visitGateId(self, ctx: ftrex_ftpParser.GateIdContext):
        return super().visitGateId(ctx)

    def visitGateType(self, ctx: ftrex_ftpParser.GateTypeContext):
        return super().visitGateType(ctx)

    def visitChildRef(self, ctx: ftrex_ftpParser.ChildRefContext):
        return super().visitChildRef(ctx)

    def visitChildRefList(self, ctx: ftrex_ftpParser.ChildRefListContext):
        return super().visitChildRefList(ctx)

    def visitMetaArgs(self, ctx: ftrex_ftpParser.MetaArgsContext):
        return super().visitMetaArgs(ctx)

    def visitMetaEncoding(self, ctx: ftrex_ftpParser.MetaEncodingContext):
        return super().visitMetaEncoding(ctx)

    def visitMetaCmd(self, ctx: ftrex_ftpParser.MetaCmdContext):
        return super().visitMetaCmd(ctx)

    def visitMetaDbName(self, ctx: ftrex_ftpParser.MetaDbNameContext):
        return super().visitMetaDbName(ctx)

    def visitMetaFTitle(self, ctx: ftrex_ftpParser.MetaFTitleContext):
        return super().visitMetaFTitle(ctx)

