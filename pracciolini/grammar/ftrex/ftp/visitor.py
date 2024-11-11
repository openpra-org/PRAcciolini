# Generated from ftrex_ftp.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .parser import ftrex_ftpParser
else:
    from parser import ftrex_ftpParser

# This class defines a complete generic visitor for a parse tree produced by ftrex_ftpParser.

class ftrex_ftpVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ftrex_ftpParser#file_.
    def visitFile_(self, ctx:ftrex_ftpParser.File_Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#section.
    def visitSection(self, ctx:ftrex_ftpParser.SectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#treeSection.
    def visitTreeSection(self, ctx:ftrex_ftpParser.TreeSectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#gate.
    def visitGate(self, ctx:ftrex_ftpParser.GateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#processSection.
    def visitProcessSection(self, ctx:ftrex_ftpParser.ProcessSectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#importSection.
    def visitImportSection(self, ctx:ftrex_ftpParser.ImportSectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#limitSection.
    def visitLimitSection(self, ctx:ftrex_ftpParser.LimitSectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#processCommands.
    def visitProcessCommands(self, ctx:ftrex_ftpParser.ProcessCommandsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#importCommands.
    def visitImportCommands(self, ctx:ftrex_ftpParser.ImportCommandsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#gateId.
    def visitGateId(self, ctx:ftrex_ftpParser.GateIdContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#gateType.
    def visitGateType(self, ctx:ftrex_ftpParser.GateTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#childRef.
    def visitChildRef(self, ctx:ftrex_ftpParser.ChildRefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#childRefList.
    def visitChildRefList(self, ctx:ftrex_ftpParser.ChildRefListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#metaArgs.
    def visitMetaArgs(self, ctx:ftrex_ftpParser.MetaArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#metaEncoding.
    def visitMetaEncoding(self, ctx:ftrex_ftpParser.MetaEncodingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#metaCmd.
    def visitMetaCmd(self, ctx:ftrex_ftpParser.MetaCmdContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#metaDbName.
    def visitMetaDbName(self, ctx:ftrex_ftpParser.MetaDbNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ftrex_ftpParser#metaFTitle.
    def visitMetaFTitle(self, ctx:ftrex_ftpParser.MetaFTitleContext):
        return self.visitChildren(ctx)



del ftrex_ftpParser