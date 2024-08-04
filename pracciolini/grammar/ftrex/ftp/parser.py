# Generated from ftrex_ftp.g4 by ANTLR 4.13.1
# encoding: utf-8
import sys
from antlr4 import ParseTreeListener, ParserRuleContext, Parser, ATNDeserializer, DFA, PredictionContextCache, Token, \
    TokenStream, ParserATNSimulator, ParseTreeVisitor, RecognitionException, NoViableAltException, ATN
from typing import TextIO


def serializedATN():
    return [
        4, 1, 14, 132, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 4, 2, 5, 7, 5, 2, 6, 7,
        6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2, 10, 7, 10, 2, 11, 7, 11, 2, 12, 7, 12, 2, 13, 7, 13,
        2, 14, 7, 14, 2, 15, 7, 15, 2, 16, 7, 16, 1, 0, 5, 0, 36, 8, 0, 10, 0, 12, 0, 39, 9, 0, 1,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 48, 8, 1, 1, 2, 5, 2, 51, 8, 2, 10, 2, 12, 2, 54,
        9, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 3, 1, 3, 4, 3, 62, 8, 3, 11, 3, 12, 3, 63, 1, 3, 1, 3, 1, 4,
        1, 4, 1, 4, 1, 5, 1, 5, 1, 5, 1, 5, 1, 6, 1, 6, 1, 6, 1, 6, 1, 7, 4, 7, 80, 8, 7, 11, 7, 12, 7,
        81, 1, 7, 4, 7, 85, 8, 7, 11, 7, 12, 7, 86, 1, 8, 1, 8, 1, 8, 4, 8, 92, 8, 8, 11, 8, 12, 8,
        93, 1, 9, 1, 9, 1, 10, 1, 10, 1, 11, 1, 11, 1, 12, 1, 12, 1, 12, 1, 12, 3, 12, 106, 8, 12,
        1, 13, 1, 13, 1, 13, 1, 14, 1, 14, 1, 14, 1, 15, 1, 15, 5, 15, 116, 8, 15, 10, 15, 12, 15,
        119, 9, 15, 1, 15, 1, 15, 1, 16, 1, 16, 5, 16, 125, 8, 16, 10, 16, 12, 16, 128, 9, 16,
        1, 16, 1, 16, 1, 16, 2, 117, 126, 0, 17, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
        26, 28, 30, 32, 0, 1, 1, 0, 5, 6, 129, 0, 37, 1, 0, 0, 0, 2, 47, 1, 0, 0, 0, 4, 52, 1, 0, 0,
        0, 6, 58, 1, 0, 0, 0, 8, 67, 1, 0, 0, 0, 10, 70, 1, 0, 0, 0, 12, 74, 1, 0, 0, 0, 14, 84, 1,
        0, 0, 0, 16, 91, 1, 0, 0, 0, 18, 95, 1, 0, 0, 0, 20, 97, 1, 0, 0, 0, 22, 99, 1, 0, 0, 0, 24,
        105, 1, 0, 0, 0, 26, 107, 1, 0, 0, 0, 28, 110, 1, 0, 0, 0, 30, 113, 1, 0, 0, 0, 32, 122,
        1, 0, 0, 0, 34, 36, 3, 2, 1, 0, 35, 34, 1, 0, 0, 0, 36, 39, 1, 0, 0, 0, 37, 35, 1, 0, 0, 0,
        37, 38, 1, 0, 0, 0, 38, 40, 1, 0, 0, 0, 39, 37, 1, 0, 0, 0, 40, 41, 5, 0, 0, 1, 41, 1, 1, 0,
        0, 0, 42, 48, 3, 4, 2, 0, 43, 48, 3, 8, 4, 0, 44, 48, 3, 10, 5, 0, 45, 48, 3, 12, 6, 0, 46,
        48, 3, 24, 12, 0, 47, 42, 1, 0, 0, 0, 47, 43, 1, 0, 0, 0, 47, 44, 1, 0, 0, 0, 47, 45, 1, 0,
        0, 0, 47, 46, 1, 0, 0, 0, 48, 3, 1, 0, 0, 0, 49, 51, 3, 6, 3, 0, 50, 49, 1, 0, 0, 0, 51, 54,
        1, 0, 0, 0, 52, 50, 1, 0, 0, 0, 52, 53, 1, 0, 0, 0, 53, 55, 1, 0, 0, 0, 54, 52, 1, 0, 0, 0,
        55, 56, 5, 1, 0, 0, 56, 57, 5, 13, 0, 0, 57, 5, 1, 0, 0, 0, 58, 59, 3, 18, 9, 0, 59, 61, 3,
        20, 10, 0, 60, 62, 3, 22, 11, 0, 61, 60, 1, 0, 0, 0, 62, 63, 1, 0, 0, 0, 63, 61, 1, 0, 0,
        0, 63, 64, 1, 0, 0, 0, 64, 65, 1, 0, 0, 0, 65, 66, 5, 13, 0, 0, 66, 7, 1, 0, 0, 0, 67, 68,
        5, 2, 0, 0, 68, 69, 3, 14, 7, 0, 69, 9, 1, 0, 0, 0, 70, 71, 5, 3, 0, 0, 71, 72, 5, 13, 0, 0,
        72, 73, 3, 16, 8, 0, 73, 11, 1, 0, 0, 0, 74, 75, 5, 4, 0, 0, 75, 76, 5, 12, 0, 0, 76, 77,
        5, 13, 0, 0, 77, 13, 1, 0, 0, 0, 78, 80, 5, 11, 0, 0, 79, 78, 1, 0, 0, 0, 80, 81, 1, 0, 0,
        0, 81, 79, 1, 0, 0, 0, 81, 82, 1, 0, 0, 0, 82, 83, 1, 0, 0, 0, 83, 85, 5, 13, 0, 0, 84, 79,
        1, 0, 0, 0, 85, 86, 1, 0, 0, 0, 86, 84, 1, 0, 0, 0, 86, 87, 1, 0, 0, 0, 87, 15, 1, 0, 0, 0,
        88, 89, 5, 12, 0, 0, 89, 90, 5, 11, 0, 0, 90, 92, 5, 13, 0, 0, 91, 88, 1, 0, 0, 0, 92, 93,
        1, 0, 0, 0, 93, 91, 1, 0, 0, 0, 93, 94, 1, 0, 0, 0, 94, 17, 1, 0, 0, 0, 95, 96, 5, 11, 0, 0,
        96, 19, 1, 0, 0, 0, 97, 98, 7, 0, 0, 0, 98, 21, 1, 0, 0, 0, 99, 100, 5, 11, 0, 0, 100, 23,
        1, 0, 0, 0, 101, 106, 3, 26, 13, 0, 102, 106, 3, 28, 14, 0, 103, 106, 3, 30, 15, 0, 104,
        106, 3, 32, 16, 0, 105, 101, 1, 0, 0, 0, 105, 102, 1, 0, 0, 0, 105, 103, 1, 0, 0, 0, 105,
        104, 1, 0, 0, 0, 106, 25, 1, 0, 0, 0, 107, 108, 5, 7, 0, 0, 108, 109, 5, 13, 0, 0, 109,
        27, 1, 0, 0, 0, 110, 111, 5, 8, 0, 0, 111, 112, 5, 13, 0, 0, 112, 29, 1, 0, 0, 0, 113, 117,
        5, 9, 0, 0, 114, 116, 9, 0, 0, 0, 115, 114, 1, 0, 0, 0, 116, 119, 1, 0, 0, 0, 117, 118,
        1, 0, 0, 0, 117, 115, 1, 0, 0, 0, 118, 120, 1, 0, 0, 0, 119, 117, 1, 0, 0, 0, 120, 121,
        5, 13, 0, 0, 121, 31, 1, 0, 0, 0, 122, 126, 5, 10, 0, 0, 123, 125, 9, 0, 0, 0, 124, 123,
        1, 0, 0, 0, 125, 128, 1, 0, 0, 0, 126, 127, 1, 0, 0, 0, 126, 124, 1, 0, 0, 0, 127, 129,
        1, 0, 0, 0, 128, 126, 1, 0, 0, 0, 129, 130, 5, 13, 0, 0, 130, 33, 1, 0, 0, 0, 10, 37, 47,
        52, 63, 81, 86, 93, 105, 117, 126
    ]


class FtrexFtpParser(Parser):
    grammarFileName = "ftrex_ftp.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]

    sharedContextCache = PredictionContextCache()

    literalNames = ["<INVALID>", "'ENDTREE'", "'PROCESS'", "'IMPORT'",
                    "'LIMIT'", "'*'", "'+'", "'**CHAR32'", "'*XEQ'", "'**DBNAME:'",
                    "'**FTITLE:'"]

    symbolicNames = ["<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                     "<INVALID>", "<INVALID>", "<INVALID>", "EVENT_ID",
                     "NUMBER", "EOL", "WS"]

    RULE_file_ = 0
    RULE_section = 1
    RULE_treeSection = 2
    RULE_gate = 3
    RULE_processSection = 4
    RULE_importSection = 5
    RULE_limitSection = 6
    RULE_processCommands = 7
    RULE_importCommands = 8
    RULE_gateId = 9
    RULE_gateType = 10
    RULE_childRef = 11
    RULE_metaArgs = 12
    RULE_metaEncoding = 13
    RULE_metaCmd = 14
    RULE_metaDbName = 15
    RULE_metaFTitle = 16

    ruleNames = ["file_", "section", "treeSection", "gate", "processSection",
                 "importSection", "limitSection", "processCommands", "importCommands",
                 "gateId", "gateType", "childRef", "metaArgs", "metaEncoding",
                 "metaCmd", "metaDbName", "metaFTitle"]

    EOF = Token.EOF
    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    EVENT_ID = 11
    NUMBER = 12
    EOL = 13
    WS = 14

    def __init__(self, input: TokenStream, output: TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None

    class File_Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(FtrexFtpParser.EOF, 0)

        def section(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(FtrexFtpParser.SectionContext)
            else:
                return self.getTypedRuleContext(FtrexFtpParser.SectionContext, i)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_file_

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterFile_"):
                listener.enterFile_(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitFile_"):
                listener.exitFile_(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitFile_"):
                return visitor.visitFile_(self)
            else:
                return visitor.visitChildren(self)

    def file_(self):

        localctx = FtrexFtpParser.File_Context(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_file_)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 37
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 3998) != 0):
                self.state = 34
                self.section()
                self.state = 39
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 40
            self.match(FtrexFtpParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def treeSection(self):
            return self.getTypedRuleContext(FtrexFtpParser.TreeSectionContext, 0)

        def processSection(self):
            return self.getTypedRuleContext(FtrexFtpParser.ProcessSectionContext, 0)

        def importSection(self):
            return self.getTypedRuleContext(FtrexFtpParser.ImportSectionContext, 0)

        def limitSection(self):
            return self.getTypedRuleContext(FtrexFtpParser.LimitSectionContext, 0)

        def metaArgs(self):
            return self.getTypedRuleContext(FtrexFtpParser.MetaArgsContext, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_section

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterSection"):
                listener.enterSection(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitSection"):
                listener.exitSection(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitSection"):
                return visitor.visitSection(self)
            else:
                return visitor.visitChildren(self)

    def section(self):

        localctx = FtrexFtpParser.SectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_section)
        try:
            self.state = 47
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [1, 11]:
                self.enterOuterAlt(localctx, 1)
                self.state = 42
                self.treeSection()
                pass
            elif token in [2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 43
                self.processSection()
                pass
            elif token in [3]:
                self.enterOuterAlt(localctx, 3)
                self.state = 44
                self.importSection()
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 4)
                self.state = 45
                self.limitSection()
                pass
            elif token in [7, 8, 9, 10]:
                self.enterOuterAlt(localctx, 5)
                self.state = 46
                self.metaArgs()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TreeSectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def gate(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(FtrexFtpParser.GateContext)
            else:
                return self.getTypedRuleContext(FtrexFtpParser.GateContext, i)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_treeSection

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterTreeSection"):
                listener.enterTreeSection(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitTreeSection"):
                listener.exitTreeSection(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitTreeSection"):
                return visitor.visitTreeSection(self)
            else:
                return visitor.visitChildren(self)

    def treeSection(self):

        localctx = FtrexFtpParser.TreeSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_treeSection)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 52
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 11:
                self.state = 49
                self.gate()
                self.state = 54
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 55
            self.match(FtrexFtpParser.T__0)
            self.state = 56
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class GateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def gateId(self):
            return self.getTypedRuleContext(FtrexFtpParser.GateIdContext, 0)

        def gateType(self):
            return self.getTypedRuleContext(FtrexFtpParser.GateTypeContext, 0)

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def childRef(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(FtrexFtpParser.ChildRefContext)
            else:
                return self.getTypedRuleContext(FtrexFtpParser.ChildRefContext, i)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_gate

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterGate"):
                listener.enterGate(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitGate"):
                listener.exitGate(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitGate"):
                return visitor.visitGate(self)
            else:
                return visitor.visitChildren(self)

    def gate(self):

        localctx = FtrexFtpParser.GateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_gate)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 58
            self.gateId()
            self.state = 59
            self.gateType()
            self.state = 61
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 60
                self.childRef()
                self.state = 63
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la == 11):
                    break

            self.state = 65
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ProcessSectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def processCommands(self):
            return self.getTypedRuleContext(FtrexFtpParser.ProcessCommandsContext, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_processSection

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterProcessSection"):
                listener.enterProcessSection(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitProcessSection"):
                listener.exitProcessSection(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitProcessSection"):
                return visitor.visitProcessSection(self)
            else:
                return visitor.visitChildren(self)

    def processSection(self):

        localctx = FtrexFtpParser.ProcessSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_processSection)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 67
            self.match(FtrexFtpParser.T__1)
            self.state = 68
            self.processCommands()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ImportSectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def importCommands(self):
            return self.getTypedRuleContext(FtrexFtpParser.ImportCommandsContext, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_importSection

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterImportSection"):
                listener.enterImportSection(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitImportSection"):
                listener.exitImportSection(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitImportSection"):
                return visitor.visitImportSection(self)
            else:
                return visitor.visitChildren(self)

    def importSection(self):

        localctx = FtrexFtpParser.ImportSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_importSection)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 70
            self.match(FtrexFtpParser.T__2)
            self.state = 71
            self.match(FtrexFtpParser.EOL)
            self.state = 72
            self.importCommands()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class LimitSectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUMBER(self):
            return self.getToken(FtrexFtpParser.NUMBER, 0)

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_limitSection

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterLimitSection"):
                listener.enterLimitSection(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitLimitSection"):
                listener.exitLimitSection(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitLimitSection"):
                return visitor.visitLimitSection(self)
            else:
                return visitor.visitChildren(self)

    def limitSection(self):

        localctx = FtrexFtpParser.LimitSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_limitSection)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 74
            self.match(FtrexFtpParser.T__3)
            self.state = 75
            self.match(FtrexFtpParser.NUMBER)
            self.state = 76
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ProcessCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self, i: int = None):
            if i is None:
                return self.getTokens(FtrexFtpParser.EOL)
            else:
                return self.getToken(FtrexFtpParser.EOL, i)

        def EVENT_ID(self, i: int = None):
            if i is None:
                return self.getTokens(FtrexFtpParser.EVENT_ID)
            else:
                return self.getToken(FtrexFtpParser.EVENT_ID, i)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_processCommands

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterProcessCommands"):
                listener.enterProcessCommands(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitProcessCommands"):
                listener.exitProcessCommands(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitProcessCommands"):
                return visitor.visitProcessCommands(self)
            else:
                return visitor.visitChildren(self)

    def processCommands(self):

        localctx = FtrexFtpParser.ProcessCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_processCommands)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self._errHandler.sync(self)
            _alt = 1
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 79
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while True:
                        self.state = 78
                        self.match(FtrexFtpParser.EVENT_ID)
                        self.state = 81
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if not (_la == 11):
                            break

                    self.state = 83
                    self.match(FtrexFtpParser.EOL)

                else:
                    raise NoViableAltException(self)
                self.state = 86
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 5, self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ImportCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUMBER(self, i: int = None):
            if i is None:
                return self.getTokens(FtrexFtpParser.NUMBER)
            else:
                return self.getToken(FtrexFtpParser.NUMBER, i)

        def EVENT_ID(self, i: int = None):
            if i is None:
                return self.getTokens(FtrexFtpParser.EVENT_ID)
            else:
                return self.getToken(FtrexFtpParser.EVENT_ID, i)

        def EOL(self, i: int = None):
            if i is None:
                return self.getTokens(FtrexFtpParser.EOL)
            else:
                return self.getToken(FtrexFtpParser.EOL, i)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_importCommands

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterImportCommands"):
                listener.enterImportCommands(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitImportCommands"):
                listener.exitImportCommands(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitImportCommands"):
                return visitor.visitImportCommands(self)
            else:
                return visitor.visitChildren(self)

    def importCommands(self):

        localctx = FtrexFtpParser.ImportCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_importCommands)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 91
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 88
                self.match(FtrexFtpParser.NUMBER)
                self.state = 89
                self.match(FtrexFtpParser.EVENT_ID)
                self.state = 90
                self.match(FtrexFtpParser.EOL)
                self.state = 93
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la == 12):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class GateIdContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EVENT_ID(self):
            return self.getToken(FtrexFtpParser.EVENT_ID, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_gateId

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterGateId"):
                listener.enterGateId(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitGateId"):
                listener.exitGateId(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitGateId"):
                return visitor.visitGateId(self)
            else:
                return visitor.visitChildren(self)

    def gateId(self):

        localctx = FtrexFtpParser.GateIdContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_gateId)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 95
            self.match(FtrexFtpParser.EVENT_ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class GateTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_gateType

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterGateType"):
                listener.enterGateType(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitGateType"):
                listener.exitGateType(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitGateType"):
                return visitor.visitGateType(self)
            else:
                return visitor.visitChildren(self)

    def gateType(self):

        localctx = FtrexFtpParser.GateTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_gateType)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 97
            _la = self._input.LA(1)
            if not (_la == 5 or _la == 6):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ChildRefContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EVENT_ID(self):
            return self.getToken(FtrexFtpParser.EVENT_ID, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_childRef

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterChildRef"):
                listener.enterChildRef(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitChildRef"):
                listener.exitChildRef(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitChildRef"):
                return visitor.visitChildRef(self)
            else:
                return visitor.visitChildren(self)

    def childRef(self):

        localctx = FtrexFtpParser.ChildRefContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_childRef)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 99
            self.match(FtrexFtpParser.EVENT_ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetaArgsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def metaEncoding(self):
            return self.getTypedRuleContext(FtrexFtpParser.MetaEncodingContext, 0)

        def metaCmd(self):
            return self.getTypedRuleContext(FtrexFtpParser.MetaCmdContext, 0)

        def metaDbName(self):
            return self.getTypedRuleContext(FtrexFtpParser.MetaDbNameContext, 0)

        def metaFTitle(self):
            return self.getTypedRuleContext(FtrexFtpParser.MetaFTitleContext, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_metaArgs

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetaArgs"):
                listener.enterMetaArgs(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetaArgs"):
                listener.exitMetaArgs(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetaArgs"):
                return visitor.visitMetaArgs(self)
            else:
                return visitor.visitChildren(self)

    def metaArgs(self):

        localctx = FtrexFtpParser.MetaArgsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_metaArgs)
        try:
            self.state = 105
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [7]:
                self.enterOuterAlt(localctx, 1)
                self.state = 101
                self.metaEncoding()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 2)
                self.state = 102
                self.metaCmd()
                pass
            elif token in [9]:
                self.enterOuterAlt(localctx, 3)
                self.state = 103
                self.metaDbName()
                pass
            elif token in [10]:
                self.enterOuterAlt(localctx, 4)
                self.state = 104
                self.metaFTitle()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetaEncodingContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_metaEncoding

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetaEncoding"):
                listener.enterMetaEncoding(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetaEncoding"):
                listener.exitMetaEncoding(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetaEncoding"):
                return visitor.visitMetaEncoding(self)
            else:
                return visitor.visitChildren(self)

    def metaEncoding(self):

        localctx = FtrexFtpParser.MetaEncodingContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_metaEncoding)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 107
            self.match(FtrexFtpParser.T__6)
            self.state = 108
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetaCmdContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_metaCmd

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetaCmd"):
                listener.enterMetaCmd(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetaCmd"):
                listener.exitMetaCmd(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetaCmd"):
                return visitor.visitMetaCmd(self)
            else:
                return visitor.visitChildren(self)

    def metaCmd(self):

        localctx = FtrexFtpParser.MetaCmdContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_metaCmd)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 110
            self.match(FtrexFtpParser.T__7)
            self.state = 111
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetaDbNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_metaDbName

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetaDbName"):
                listener.enterMetaDbName(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetaDbName"):
                listener.exitMetaDbName(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetaDbName"):
                return visitor.visitMetaDbName(self)
            else:
                return visitor.visitChildren(self)

    def metaDbName(self):

        localctx = FtrexFtpParser.MetaDbNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_metaDbName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 113
            self.match(FtrexFtpParser.T__8)
            self.state = 117
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 8, self._ctx)
            while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1 + 1:
                    self.state = 114
                    self.matchWildcard()
                self.state = 119
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 8, self._ctx)

            self.state = 120
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MetaFTitleContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(FtrexFtpParser.EOL, 0)

        def getRuleIndex(self):
            return FtrexFtpParser.RULE_metaFTitle

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMetaFTitle"):
                listener.enterMetaFTitle(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMetaFTitle"):
                listener.exitMetaFTitle(self)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, "visitMetaFTitle"):
                return visitor.visitMetaFTitle(self)
            else:
                return visitor.visitChildren(self)

    def metaFTitle(self):

        localctx = FtrexFtpParser.MetaFTitleContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_metaFTitle)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 122
            self.match(FtrexFtpParser.T__9)
            self.state = 126
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 9, self._ctx)
            while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1 + 1:
                    self.state = 123
                    self.matchWildcard()
                self.state = 128
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 9, self._ctx)

            self.state = 129
            self.match(FtrexFtpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx
