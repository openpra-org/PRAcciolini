# Generated from ftrex_ftp.g4 by ANTLR 4.13.1
from antlr4 import ParseTreeListener

from pracciolini.grammar.ftrex.ftp.parser import FtrexFtpParser


# This class defines a complete listener for a parse tree produced by FtrexFtpParser.
class FtrexFtpListener(ParseTreeListener):

    # Enter a parse tree produced by FtrexFtpParser#file_.
    def enterFile_(self, ctx: FtrexFtpParser.File_Context):
        pass

    # Exit a parse tree produced by FtrexFtpParser#file_.
    def exitFile_(self, ctx: FtrexFtpParser.File_Context):
        pass

    # Enter a parse tree produced by FtrexFtpParser#section.
    def enterSection(self, ctx: FtrexFtpParser.SectionContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#section.
    def exitSection(self, ctx: FtrexFtpParser.SectionContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#treeSection.
    def enterTreeSection(self, ctx: FtrexFtpParser.TreeSectionContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#treeSection.
    def exitTreeSection(self, ctx: FtrexFtpParser.TreeSectionContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#gate.
    def enterGate(self, ctx: FtrexFtpParser.GateContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#gate.
    def exitGate(self, ctx: FtrexFtpParser.GateContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#processSection.
    def enterProcessSection(self, ctx: FtrexFtpParser.ProcessSectionContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#processSection.
    def exitProcessSection(self, ctx: FtrexFtpParser.ProcessSectionContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#importSection.
    def enterImportSection(self, ctx: FtrexFtpParser.ImportSectionContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#importSection.
    def exitImportSection(self, ctx: FtrexFtpParser.ImportSectionContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#limitSection.
    def enterLimitSection(self, ctx: FtrexFtpParser.LimitSectionContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#limitSection.
    def exitLimitSection(self, ctx: FtrexFtpParser.LimitSectionContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#processCommands.
    def enterProcessCommands(self, ctx: FtrexFtpParser.ProcessCommandsContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#processCommands.
    def exitProcessCommands(self, ctx: FtrexFtpParser.ProcessCommandsContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#importCommands.
    def enterImportCommands(self, ctx: FtrexFtpParser.ImportCommandsContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#importCommands.
    def exitImportCommands(self, ctx: FtrexFtpParser.ImportCommandsContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#gateId.
    def enterGateId(self, ctx: FtrexFtpParser.GateIdContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#gateId.
    def exitGateId(self, ctx: FtrexFtpParser.GateIdContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#gateType.
    def enterGateType(self, ctx: FtrexFtpParser.GateTypeContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#gateType.
    def exitGateType(self, ctx: FtrexFtpParser.GateTypeContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#childRef.
    def enterChildRef(self, ctx: FtrexFtpParser.ChildRefContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#childRef.
    def exitChildRef(self, ctx: FtrexFtpParser.ChildRefContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#metaArgs.
    def enterMetaArgs(self, ctx: FtrexFtpParser.MetaArgsContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#metaArgs.
    def exitMetaArgs(self, ctx: FtrexFtpParser.MetaArgsContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#metaEncoding.
    def enterMetaEncoding(self, ctx: FtrexFtpParser.MetaEncodingContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#metaEncoding.
    def exitMetaEncoding(self, ctx: FtrexFtpParser.MetaEncodingContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#metaCmd.
    def enterMetaCmd(self, ctx: FtrexFtpParser.MetaCmdContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#metaCmd.
    def exitMetaCmd(self, ctx: FtrexFtpParser.MetaCmdContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#metaDbName.
    def enterMetaDbName(self, ctx: FtrexFtpParser.MetaDbNameContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#metaDbName.
    def exitMetaDbName(self, ctx: FtrexFtpParser.MetaDbNameContext):
        pass

    # Enter a parse tree produced by FtrexFtpParser#metaFTitle.
    def enterMetaFTitle(self, ctx: FtrexFtpParser.MetaFTitleContext):
        pass

    # Exit a parse tree produced by FtrexFtpParser#metaFTitle.
    def exitMetaFTitle(self, ctx: FtrexFtpParser.MetaFTitleContext):
        pass


del FtrexFtpParser
