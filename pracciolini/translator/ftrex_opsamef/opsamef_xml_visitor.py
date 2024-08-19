from lxml import etree

from pracciolini.grammar.ftrex.ftp.parser import ftrex_ftpParser
from pracciolini.grammar.ftrex.ftp.visitor import ftrex_ftpVisitor


class OpsaMefXmlVisitor(ftrex_ftpVisitor):
    def __init__(self):
        self.xml_doc: etree.ElementTree = etree.ElementTree(etree.Element("opsa-mef"))
        super().__init__()

    def build_xml(self, tree):
        parsed = self.visit(tree)
        return self.xml_doc

    @staticmethod
    def create_basic_event(name, label = None, probability = None):
        basic_event = etree.Element("define-basic-event", name=name)
        if probability is not None:
            probability_elem = etree.SubElement(basic_event, "float", value=str(probability))
        if label is not None:
            label_elem = etree.SubElement(basic_event, "label")
            label_elem.text = str(label)
        return basic_event

    def visitFile_(self, ctx: ftrex_ftpParser.File_Context):
        print("visitFile_", ctx.getText())
        return super().visitFile_(ctx)

    def visitSection(self, ctx: ftrex_ftpParser.SectionContext):
        print("visitSection", ctx.getText())
        return super().visitSection(ctx)

    def visitTreeSection(self, ctx: ftrex_ftpParser.TreeSectionContext):
        print("visitTreeSection", ctx.getText())
        return super().visitTreeSection(ctx)

    def visitGate(self, ctx: ftrex_ftpParser.GateContext):
        return super().visitGate(ctx)

    def visitProcessSection(self, ctx: ftrex_ftpParser.ProcessSectionContext):
        return super().visitProcessSection(ctx)

    def visitImportSection(self, ctx: ftrex_ftpParser.ImportSectionContext):
        return super().visitImportSection(ctx)

    def visitLimitSection(self, ctx: ftrex_ftpParser.LimitSectionContext):
        return super().visitLimitSection(ctx)

    def visitProcessCommands(self, ctx: ftrex_ftpParser.ProcessCommandsContext):
        return super().visitProcessCommands(ctx)

    def visitImportCommands(self, ctx: ftrex_ftpParser.ImportCommandsContext):
        print(ctx.getText(), ctx.EVENT_ID())
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
