# Generated from ftrex_ftp.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .parser import ftrex_ftpParser
else:
    from parser import ftrex_ftpParser

# This class defines a complete listener for a parse tree produced by ftrex_ftpParser.
class ftrex_ftpListener(ParseTreeListener):

    # Enter a parse tree produced by ftrex_ftpParser#file_.
    def enterFile_(self, ctx:ftrex_ftpParser.File_Context):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#file_.
    def exitFile_(self, ctx:ftrex_ftpParser.File_Context):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#section.
    def enterSection(self, ctx:ftrex_ftpParser.SectionContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#section.
    def exitSection(self, ctx:ftrex_ftpParser.SectionContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#treeSection.
    def enterTreeSection(self, ctx:ftrex_ftpParser.TreeSectionContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#treeSection.
    def exitTreeSection(self, ctx:ftrex_ftpParser.TreeSectionContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#gate.
    def enterGate(self, ctx:ftrex_ftpParser.GateContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#gate.
    def exitGate(self, ctx:ftrex_ftpParser.GateContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#processSection.
    def enterProcessSection(self, ctx:ftrex_ftpParser.ProcessSectionContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#processSection.
    def exitProcessSection(self, ctx:ftrex_ftpParser.ProcessSectionContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#importSection.
    def enterImportSection(self, ctx:ftrex_ftpParser.ImportSectionContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#importSection.
    def exitImportSection(self, ctx:ftrex_ftpParser.ImportSectionContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#limitSection.
    def enterLimitSection(self, ctx:ftrex_ftpParser.LimitSectionContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#limitSection.
    def exitLimitSection(self, ctx:ftrex_ftpParser.LimitSectionContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#processCommands.
    def enterProcessCommands(self, ctx:ftrex_ftpParser.ProcessCommandsContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#processCommands.
    def exitProcessCommands(self, ctx:ftrex_ftpParser.ProcessCommandsContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#importCommands.
    def enterImportCommands(self, ctx:ftrex_ftpParser.ImportCommandsContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#importCommands.
    def exitImportCommands(self, ctx:ftrex_ftpParser.ImportCommandsContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#gateId.
    def enterGateId(self, ctx:ftrex_ftpParser.GateIdContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#gateId.
    def exitGateId(self, ctx:ftrex_ftpParser.GateIdContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#gateType.
    def enterGateType(self, ctx:ftrex_ftpParser.GateTypeContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#gateType.
    def exitGateType(self, ctx:ftrex_ftpParser.GateTypeContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#childRef.
    def enterChildRef(self, ctx:ftrex_ftpParser.ChildRefContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#childRef.
    def exitChildRef(self, ctx:ftrex_ftpParser.ChildRefContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#childRefList.
    def enterChildRefList(self, ctx:ftrex_ftpParser.ChildRefListContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#childRefList.
    def exitChildRefList(self, ctx:ftrex_ftpParser.ChildRefListContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#metaArgs.
    def enterMetaArgs(self, ctx:ftrex_ftpParser.MetaArgsContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#metaArgs.
    def exitMetaArgs(self, ctx:ftrex_ftpParser.MetaArgsContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#metaEncoding.
    def enterMetaEncoding(self, ctx:ftrex_ftpParser.MetaEncodingContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#metaEncoding.
    def exitMetaEncoding(self, ctx:ftrex_ftpParser.MetaEncodingContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#metaCmd.
    def enterMetaCmd(self, ctx:ftrex_ftpParser.MetaCmdContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#metaCmd.
    def exitMetaCmd(self, ctx:ftrex_ftpParser.MetaCmdContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#metaDbName.
    def enterMetaDbName(self, ctx:ftrex_ftpParser.MetaDbNameContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#metaDbName.
    def exitMetaDbName(self, ctx:ftrex_ftpParser.MetaDbNameContext):
        pass


    # Enter a parse tree produced by ftrex_ftpParser#metaFTitle.
    def enterMetaFTitle(self, ctx:ftrex_ftpParser.MetaFTitleContext):
        pass

    # Exit a parse tree produced by ftrex_ftpParser#metaFTitle.
    def exitMetaFTitle(self, ctx:ftrex_ftpParser.MetaFTitleContext):
        pass



del ftrex_ftpParser