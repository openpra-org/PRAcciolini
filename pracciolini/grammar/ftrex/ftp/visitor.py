# Generated from ftrex_ftp.g4 by ANTLR 4.13.1
from antlr4 import ParseTreeVisitor

from pracciolini.grammar.ftrex.ftp.parser import FtrexFtpParser


# This class defines a complete generic visitor for a parse tree produced by FtrexFtpParser.

class FtrexFtpVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by FtrexFtpParser#file_.
    def visitFile_(self, ctx: FtrexFtpParser.File_Context):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#section.
    def visitSection(self, ctx: FtrexFtpParser.SectionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#treeSection.
    def visitTreeSection(self, ctx: FtrexFtpParser.TreeSectionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#gate.
    def visitGate(self, ctx: FtrexFtpParser.GateContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#processSection.
    def visitProcessSection(self, ctx: FtrexFtpParser.ProcessSectionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#importSection.
    def visitImportSection(self, ctx: FtrexFtpParser.ImportSectionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#limitSection.
    def visitLimitSection(self, ctx: FtrexFtpParser.LimitSectionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#processCommands.
    def visitProcessCommands(self, ctx: FtrexFtpParser.ProcessCommandsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#importCommands.
    def visitImportCommands(self, ctx: FtrexFtpParser.ImportCommandsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#gateId.
    def visitGateId(self, ctx: FtrexFtpParser.GateIdContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#gateType.
    def visitGateType(self, ctx: FtrexFtpParser.GateTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#childRef.
    def visitChildRef(self, ctx: FtrexFtpParser.ChildRefContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#metaArgs.
    def visitMetaArgs(self, ctx: FtrexFtpParser.MetaArgsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#metaEncoding.
    def visitMetaEncoding(self, ctx: FtrexFtpParser.MetaEncodingContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#metaCmd.
    def visitMetaCmd(self, ctx: FtrexFtpParser.MetaCmdContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#metaDbName.
    def visitMetaDbName(self, ctx: FtrexFtpParser.MetaDbNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by FtrexFtpParser#metaFTitle.
    def visitMetaFTitle(self, ctx: FtrexFtpParser.MetaFTitleContext):
        return self.visitChildren(ctx)


del FtrexFtpParser
