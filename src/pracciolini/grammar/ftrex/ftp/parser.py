# Generated from ftrex_ftp.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import Parser, ATNDeserializer

import sys

from antlr4.BufferedTokenStream import TokenStream, Token
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.PredictionContext import PredictionContextCache
from antlr4.atn.ATN import ATN
from antlr4.atn.ParserATNSimulator import ParserATNSimulator
from antlr4.dfa.DFA import DFA
from antlr4.error.Errors import NoViableAltException, RecognitionException
from antlr4.tree.Tree import ParseTreeListener, ParseTreeVisitor

from typing import TextIO

def serializedATN():
    return [
        4,1,18,172,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,1,0,5,0,50,8,0,10,0,12,0,53,9,
        0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,3,1,62,8,1,1,2,5,2,65,8,2,10,2,12,
        2,68,9,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,4,1,4,1,5,1,5,1,5,1,6,1,6,
        1,7,3,7,85,8,7,1,7,1,7,1,7,5,7,90,8,7,10,7,12,7,93,9,7,1,8,1,8,3,
        8,97,8,8,1,9,1,9,1,9,1,10,1,10,1,11,1,11,1,11,1,12,1,12,1,12,1,12,
        1,13,1,13,1,13,1,13,1,14,4,14,116,8,14,11,14,12,14,117,1,14,4,14,
        121,8,14,11,14,12,14,122,1,15,1,15,3,15,127,8,15,1,15,1,15,4,15,
        131,8,15,11,15,12,15,132,1,16,1,16,1,16,1,17,1,17,1,18,1,18,1,19,
        1,19,1,19,1,19,3,19,146,8,19,1,20,1,20,1,20,1,21,1,21,1,21,1,22,
        1,22,5,22,156,8,22,10,22,12,22,159,9,22,1,22,1,22,1,23,1,23,5,23,
        165,8,23,10,23,12,23,168,9,23,1,23,1,23,1,23,2,157,166,0,24,0,2,
        4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,0,
        1,1,0,10,12,166,0,51,1,0,0,0,2,61,1,0,0,0,4,66,1,0,0,0,6,72,1,0,
        0,0,8,76,1,0,0,0,10,78,1,0,0,0,12,81,1,0,0,0,14,84,1,0,0,0,16,96,
        1,0,0,0,18,98,1,0,0,0,20,101,1,0,0,0,22,103,1,0,0,0,24,106,1,0,0,
        0,26,110,1,0,0,0,28,120,1,0,0,0,30,130,1,0,0,0,32,134,1,0,0,0,34,
        137,1,0,0,0,36,139,1,0,0,0,38,145,1,0,0,0,40,147,1,0,0,0,42,150,
        1,0,0,0,44,153,1,0,0,0,46,162,1,0,0,0,48,50,3,2,1,0,49,48,1,0,0,
        0,50,53,1,0,0,0,51,49,1,0,0,0,51,52,1,0,0,0,52,54,1,0,0,0,53,51,
        1,0,0,0,54,55,5,0,0,1,55,1,1,0,0,0,56,62,3,4,2,0,57,62,3,22,11,0,
        58,62,3,24,12,0,59,62,3,26,13,0,60,62,3,38,19,0,61,56,1,0,0,0,61,
        57,1,0,0,0,61,58,1,0,0,0,61,59,1,0,0,0,61,60,1,0,0,0,62,3,1,0,0,
        0,63,65,3,6,3,0,64,63,1,0,0,0,65,68,1,0,0,0,66,64,1,0,0,0,66,67,
        1,0,0,0,67,69,1,0,0,0,68,66,1,0,0,0,69,70,5,1,0,0,70,71,5,17,0,0,
        71,5,1,0,0,0,72,73,3,8,4,0,73,74,3,10,5,0,74,75,5,17,0,0,75,7,1,
        0,0,0,76,77,5,16,0,0,77,9,1,0,0,0,78,79,3,12,6,0,79,80,3,14,7,0,
        80,11,1,0,0,0,81,82,7,0,0,0,82,13,1,0,0,0,83,85,5,17,0,0,84,83,1,
        0,0,0,84,85,1,0,0,0,85,86,1,0,0,0,86,91,3,16,8,0,87,90,3,16,8,0,
        88,90,5,17,0,0,89,87,1,0,0,0,89,88,1,0,0,0,90,93,1,0,0,0,91,89,1,
        0,0,0,91,92,1,0,0,0,92,15,1,0,0,0,93,91,1,0,0,0,94,97,3,20,10,0,
        95,97,3,18,9,0,96,94,1,0,0,0,96,95,1,0,0,0,97,17,1,0,0,0,98,99,5,
        2,0,0,99,100,3,20,10,0,100,19,1,0,0,0,101,102,5,16,0,0,102,21,1,
        0,0,0,103,104,5,3,0,0,104,105,3,28,14,0,105,23,1,0,0,0,106,107,5,
        4,0,0,107,108,5,17,0,0,108,109,3,30,15,0,109,25,1,0,0,0,110,111,
        5,5,0,0,111,112,5,15,0,0,112,113,5,17,0,0,113,27,1,0,0,0,114,116,
        5,16,0,0,115,114,1,0,0,0,116,117,1,0,0,0,117,115,1,0,0,0,117,118,
        1,0,0,0,118,119,1,0,0,0,119,121,5,17,0,0,120,115,1,0,0,0,121,122,
        1,0,0,0,122,120,1,0,0,0,122,123,1,0,0,0,123,29,1,0,0,0,124,126,3,
        32,16,0,125,127,5,14,0,0,126,125,1,0,0,0,126,127,1,0,0,0,127,128,
        1,0,0,0,128,129,5,17,0,0,129,131,1,0,0,0,130,124,1,0,0,0,131,132,
        1,0,0,0,132,130,1,0,0,0,132,133,1,0,0,0,133,31,1,0,0,0,134,135,3,
        36,18,0,135,136,3,34,17,0,136,33,1,0,0,0,137,138,5,16,0,0,138,35,
        1,0,0,0,139,140,5,15,0,0,140,37,1,0,0,0,141,146,3,40,20,0,142,146,
        3,42,21,0,143,146,3,44,22,0,144,146,3,46,23,0,145,141,1,0,0,0,145,
        142,1,0,0,0,145,143,1,0,0,0,145,144,1,0,0,0,146,39,1,0,0,0,147,148,
        5,6,0,0,148,149,5,17,0,0,149,41,1,0,0,0,150,151,5,7,0,0,151,152,
        5,17,0,0,152,43,1,0,0,0,153,157,5,8,0,0,154,156,9,0,0,0,155,154,
        1,0,0,0,156,159,1,0,0,0,157,158,1,0,0,0,157,155,1,0,0,0,158,160,
        1,0,0,0,159,157,1,0,0,0,160,161,5,17,0,0,161,45,1,0,0,0,162,166,
        5,9,0,0,163,165,9,0,0,0,164,163,1,0,0,0,165,168,1,0,0,0,166,167,
        1,0,0,0,166,164,1,0,0,0,167,169,1,0,0,0,168,166,1,0,0,0,169,170,
        5,17,0,0,170,47,1,0,0,0,14,51,61,66,84,89,91,96,117,122,126,132,
        145,157,166
    ]

class ftrex_ftpParser ( Parser ):

    grammarFileName = "ftrex_ftp.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'ENDTREE'", "'-'", "'PROCESS'", "'IMPORT'",
                     "'LIMIT'", "'**CHAR32'", "'*XEQ'", "'**DBNAME:'", "'**FTITLE:'",
                     "'*'", "'+'", "<INVALID>", "<INVALID>", "'I'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "AND", "OR", "ATLEAST",
                      "NON_NEG_INTEGER", "INITIATOR_TAG", "REAL_NUMBER",
                      "EVENT_ID", "EOL", "WS" ]

    RULE_file_ = 0
    RULE_section = 1
    RULE_treeSection = 2
    RULE_gate = 3
    RULE_gateId = 4
    RULE_gateDef = 5
    RULE_gateType = 6
    RULE_operands = 7
    RULE_literal = 8
    RULE_notEvent = 9
    RULE_event = 10
    RULE_processSection = 11
    RULE_importSection = 12
    RULE_limitSection = 13
    RULE_processCommands = 14
    RULE_importCommands = 15
    RULE_basicEvent = 16
    RULE_basicEventID = 17
    RULE_probability = 18
    RULE_metaArgs = 19
    RULE_metaEncoding = 20
    RULE_metaCmd = 21
    RULE_metaDbName = 22
    RULE_metaFTitle = 23

    ruleNames =  [ "file_", "section", "treeSection", "gate", "gateId",
                   "gateDef", "gateType", "operands", "literal", "notEvent",
                   "event", "processSection", "importSection", "limitSection",
                   "processCommands", "importCommands", "basicEvent", "basicEventID",
                   "probability", "metaArgs", "metaEncoding", "metaCmd",
                   "metaDbName", "metaFTitle" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    AND=10
    OR=11
    ATLEAST=12
    NON_NEG_INTEGER=13
    INITIATOR_TAG=14
    REAL_NUMBER=15
    EVENT_ID=16
    EOL=17
    WS=18

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class File_Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(ftrex_ftpParser.EOF, 0)

        def section(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ftrex_ftpParser.SectionContext)
            else:
                return self.getTypedRuleContext(ftrex_ftpParser.SectionContext,i)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_file_

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFile_" ):
                listener.enterFile_(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFile_" ):
                listener.exitFile_(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFile_" ):
                return visitor.visitFile_(self)
            else:
                return visitor.visitChildren(self)




    def file_(self):

        localctx = ftrex_ftpParser.File_Context(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_file_)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 51
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 66554) != 0):
                self.state = 48
                self.section()
                self.state = 53
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 54
            self.match(ftrex_ftpParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def treeSection(self):
            return self.getTypedRuleContext(ftrex_ftpParser.TreeSectionContext,0)


        def processSection(self):
            return self.getTypedRuleContext(ftrex_ftpParser.ProcessSectionContext,0)


        def importSection(self):
            return self.getTypedRuleContext(ftrex_ftpParser.ImportSectionContext,0)


        def limitSection(self):
            return self.getTypedRuleContext(ftrex_ftpParser.LimitSectionContext,0)


        def metaArgs(self):
            return self.getTypedRuleContext(ftrex_ftpParser.MetaArgsContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_section

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSection" ):
                listener.enterSection(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSection" ):
                listener.exitSection(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSection" ):
                return visitor.visitSection(self)
            else:
                return visitor.visitChildren(self)




    def section(self):

        localctx = ftrex_ftpParser.SectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_section)
        try:
            self.state = 61
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [1, 16]:
                self.enterOuterAlt(localctx, 1)
                self.state = 56
                self.treeSection()
                pass
            elif token in [3]:
                self.enterOuterAlt(localctx, 2)
                self.state = 57
                self.processSection()
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 3)
                self.state = 58
                self.importSection()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 4)
                self.state = 59
                self.limitSection()
                pass
            elif token in [6, 7, 8, 9]:
                self.enterOuterAlt(localctx, 5)
                self.state = 60
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def gate(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ftrex_ftpParser.GateContext)
            else:
                return self.getTypedRuleContext(ftrex_ftpParser.GateContext,i)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_treeSection

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTreeSection" ):
                listener.enterTreeSection(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTreeSection" ):
                listener.exitTreeSection(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTreeSection" ):
                return visitor.visitTreeSection(self)
            else:
                return visitor.visitChildren(self)




    def treeSection(self):

        localctx = ftrex_ftpParser.TreeSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_treeSection)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 66
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==16:
                self.state = 63
                self.gate()
                self.state = 68
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 69
            self.match(ftrex_ftpParser.T__0)
            self.state = 70
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def gateId(self):
            return self.getTypedRuleContext(ftrex_ftpParser.GateIdContext,0)


        def gateDef(self):
            return self.getTypedRuleContext(ftrex_ftpParser.GateDefContext,0)


        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_gate

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGate" ):
                listener.enterGate(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGate" ):
                listener.exitGate(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGate" ):
                return visitor.visitGate(self)
            else:
                return visitor.visitChildren(self)




    def gate(self):

        localctx = ftrex_ftpParser.GateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_gate)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            self.gateId()
            self.state = 73
            self.gateDef()
            self.state = 74
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GateIdContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EVENT_ID(self):
            return self.getToken(ftrex_ftpParser.EVENT_ID, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_gateId

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGateId" ):
                listener.enterGateId(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGateId" ):
                listener.exitGateId(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGateId" ):
                return visitor.visitGateId(self)
            else:
                return visitor.visitChildren(self)




    def gateId(self):

        localctx = ftrex_ftpParser.GateIdContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_gateId)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 76
            self.match(ftrex_ftpParser.EVENT_ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GateDefContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def gateType(self):
            return self.getTypedRuleContext(ftrex_ftpParser.GateTypeContext,0)


        def operands(self):
            return self.getTypedRuleContext(ftrex_ftpParser.OperandsContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_gateDef

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGateDef" ):
                listener.enterGateDef(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGateDef" ):
                listener.exitGateDef(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGateDef" ):
                return visitor.visitGateDef(self)
            else:
                return visitor.visitChildren(self)




    def gateDef(self):

        localctx = ftrex_ftpParser.GateDefContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_gateDef)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 78
            self.gateType()
            self.state = 79
            self.operands()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GateTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def AND(self):
            return self.getToken(ftrex_ftpParser.AND, 0)

        def OR(self):
            return self.getToken(ftrex_ftpParser.OR, 0)

        def ATLEAST(self):
            return self.getToken(ftrex_ftpParser.ATLEAST, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_gateType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGateType" ):
                listener.enterGateType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGateType" ):
                listener.exitGateType(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGateType" ):
                return visitor.visitGateType(self)
            else:
                return visitor.visitChildren(self)




    def gateType(self):

        localctx = ftrex_ftpParser.GateTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_gateType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 81
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 7168) != 0)):
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


    class OperandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def literal(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ftrex_ftpParser.LiteralContext)
            else:
                return self.getTypedRuleContext(ftrex_ftpParser.LiteralContext,i)


        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(ftrex_ftpParser.EOL)
            else:
                return self.getToken(ftrex_ftpParser.EOL, i)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_operands

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOperands" ):
                listener.enterOperands(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOperands" ):
                listener.exitOperands(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOperands" ):
                return visitor.visitOperands(self)
            else:
                return visitor.visitChildren(self)




    def operands(self):

        localctx = ftrex_ftpParser.OperandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_operands)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==17:
                self.state = 83
                self.match(ftrex_ftpParser.EOL)


            self.state = 86
            self.literal()
            self.state = 91
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,5,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 89
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [2, 16]:
                        self.state = 87
                        self.literal()
                        pass
                    elif token in [17]:
                        self.state = 88
                        self.match(ftrex_ftpParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)

                self.state = 93
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LiteralContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def event(self):
            return self.getTypedRuleContext(ftrex_ftpParser.EventContext,0)


        def notEvent(self):
            return self.getTypedRuleContext(ftrex_ftpParser.NotEventContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_literal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLiteral" ):
                listener.enterLiteral(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLiteral" ):
                listener.exitLiteral(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLiteral" ):
                return visitor.visitLiteral(self)
            else:
                return visitor.visitChildren(self)




    def literal(self):

        localctx = ftrex_ftpParser.LiteralContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_literal)
        try:
            self.state = 96
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [16]:
                self.enterOuterAlt(localctx, 1)
                self.state = 94
                self.event()
                pass
            elif token in [2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 95
                self.notEvent()
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


    class NotEventContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def event(self):
            return self.getTypedRuleContext(ftrex_ftpParser.EventContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_notEvent

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNotEvent" ):
                listener.enterNotEvent(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNotEvent" ):
                listener.exitNotEvent(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNotEvent" ):
                return visitor.visitNotEvent(self)
            else:
                return visitor.visitChildren(self)




    def notEvent(self):

        localctx = ftrex_ftpParser.NotEventContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_notEvent)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 98
            self.match(ftrex_ftpParser.T__1)
            self.state = 99
            self.event()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EventContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EVENT_ID(self):
            return self.getToken(ftrex_ftpParser.EVENT_ID, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_event

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEvent" ):
                listener.enterEvent(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEvent" ):
                listener.exitEvent(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEvent" ):
                return visitor.visitEvent(self)
            else:
                return visitor.visitChildren(self)




    def event(self):

        localctx = ftrex_ftpParser.EventContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_event)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self.match(ftrex_ftpParser.EVENT_ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ProcessSectionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def processCommands(self):
            return self.getTypedRuleContext(ftrex_ftpParser.ProcessCommandsContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_processSection

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProcessSection" ):
                listener.enterProcessSection(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProcessSection" ):
                listener.exitProcessSection(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProcessSection" ):
                return visitor.visitProcessSection(self)
            else:
                return visitor.visitChildren(self)




    def processSection(self):

        localctx = ftrex_ftpParser.ProcessSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_processSection)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 103
            self.match(ftrex_ftpParser.T__2)
            self.state = 104
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def importCommands(self):
            return self.getTypedRuleContext(ftrex_ftpParser.ImportCommandsContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_importSection

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterImportSection" ):
                listener.enterImportSection(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitImportSection" ):
                listener.exitImportSection(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitImportSection" ):
                return visitor.visitImportSection(self)
            else:
                return visitor.visitChildren(self)




    def importSection(self):

        localctx = ftrex_ftpParser.ImportSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_importSection)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 106
            self.match(ftrex_ftpParser.T__3)
            self.state = 107
            self.match(ftrex_ftpParser.EOL)
            self.state = 108
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REAL_NUMBER(self):
            return self.getToken(ftrex_ftpParser.REAL_NUMBER, 0)

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_limitSection

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLimitSection" ):
                listener.enterLimitSection(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLimitSection" ):
                listener.exitLimitSection(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLimitSection" ):
                return visitor.visitLimitSection(self)
            else:
                return visitor.visitChildren(self)




    def limitSection(self):

        localctx = ftrex_ftpParser.LimitSectionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_limitSection)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 110
            self.match(ftrex_ftpParser.T__4)
            self.state = 111
            self.match(ftrex_ftpParser.REAL_NUMBER)
            self.state = 112
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ProcessCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(ftrex_ftpParser.EOL)
            else:
                return self.getToken(ftrex_ftpParser.EOL, i)

        def EVENT_ID(self, i:int=None):
            if i is None:
                return self.getTokens(ftrex_ftpParser.EVENT_ID)
            else:
                return self.getToken(ftrex_ftpParser.EVENT_ID, i)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_processCommands

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProcessCommands" ):
                listener.enterProcessCommands(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProcessCommands" ):
                listener.exitProcessCommands(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProcessCommands" ):
                return visitor.visitProcessCommands(self)
            else:
                return visitor.visitChildren(self)




    def processCommands(self):

        localctx = ftrex_ftpParser.ProcessCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_processCommands)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 120
            self._errHandler.sync(self)
            _alt = 1
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 115
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while True:
                        self.state = 114
                        self.match(ftrex_ftpParser.EVENT_ID)
                        self.state = 117
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if not (_la==16):
                            break

                    self.state = 119
                    self.match(ftrex_ftpParser.EOL)

                else:
                    raise NoViableAltException(self)
                self.state = 122
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ImportCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def basicEvent(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ftrex_ftpParser.BasicEventContext)
            else:
                return self.getTypedRuleContext(ftrex_ftpParser.BasicEventContext,i)


        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(ftrex_ftpParser.EOL)
            else:
                return self.getToken(ftrex_ftpParser.EOL, i)

        def INITIATOR_TAG(self, i:int=None):
            if i is None:
                return self.getTokens(ftrex_ftpParser.INITIATOR_TAG)
            else:
                return self.getToken(ftrex_ftpParser.INITIATOR_TAG, i)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_importCommands

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterImportCommands" ):
                listener.enterImportCommands(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitImportCommands" ):
                listener.exitImportCommands(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitImportCommands" ):
                return visitor.visitImportCommands(self)
            else:
                return visitor.visitChildren(self)




    def importCommands(self):

        localctx = ftrex_ftpParser.ImportCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_importCommands)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 130
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 124
                self.basicEvent()
                self.state = 126
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==14:
                    self.state = 125
                    self.match(ftrex_ftpParser.INITIATOR_TAG)


                self.state = 128
                self.match(ftrex_ftpParser.EOL)
                self.state = 132
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==15):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BasicEventContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def probability(self):
            return self.getTypedRuleContext(ftrex_ftpParser.ProbabilityContext,0)


        def basicEventID(self):
            return self.getTypedRuleContext(ftrex_ftpParser.BasicEventIDContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_basicEvent

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBasicEvent" ):
                listener.enterBasicEvent(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBasicEvent" ):
                listener.exitBasicEvent(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBasicEvent" ):
                return visitor.visitBasicEvent(self)
            else:
                return visitor.visitChildren(self)




    def basicEvent(self):

        localctx = ftrex_ftpParser.BasicEventContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_basicEvent)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 134
            self.probability()
            self.state = 135
            self.basicEventID()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BasicEventIDContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EVENT_ID(self):
            return self.getToken(ftrex_ftpParser.EVENT_ID, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_basicEventID

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBasicEventID" ):
                listener.enterBasicEventID(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBasicEventID" ):
                listener.exitBasicEventID(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBasicEventID" ):
                return visitor.visitBasicEventID(self)
            else:
                return visitor.visitChildren(self)




    def basicEventID(self):

        localctx = ftrex_ftpParser.BasicEventIDContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_basicEventID)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 137
            self.match(ftrex_ftpParser.EVENT_ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ProbabilityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REAL_NUMBER(self):
            return self.getToken(ftrex_ftpParser.REAL_NUMBER, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_probability

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProbability" ):
                listener.enterProbability(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProbability" ):
                listener.exitProbability(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProbability" ):
                return visitor.visitProbability(self)
            else:
                return visitor.visitChildren(self)




    def probability(self):

        localctx = ftrex_ftpParser.ProbabilityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_probability)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 139
            self.match(ftrex_ftpParser.REAL_NUMBER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MetaArgsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def metaEncoding(self):
            return self.getTypedRuleContext(ftrex_ftpParser.MetaEncodingContext,0)


        def metaCmd(self):
            return self.getTypedRuleContext(ftrex_ftpParser.MetaCmdContext,0)


        def metaDbName(self):
            return self.getTypedRuleContext(ftrex_ftpParser.MetaDbNameContext,0)


        def metaFTitle(self):
            return self.getTypedRuleContext(ftrex_ftpParser.MetaFTitleContext,0)


        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_metaArgs

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMetaArgs" ):
                listener.enterMetaArgs(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMetaArgs" ):
                listener.exitMetaArgs(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaArgs" ):
                return visitor.visitMetaArgs(self)
            else:
                return visitor.visitChildren(self)




    def metaArgs(self):

        localctx = ftrex_ftpParser.MetaArgsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_metaArgs)
        try:
            self.state = 145
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [6]:
                self.enterOuterAlt(localctx, 1)
                self.state = 141
                self.metaEncoding()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 2)
                self.state = 142
                self.metaCmd()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 3)
                self.state = 143
                self.metaDbName()
                pass
            elif token in [9]:
                self.enterOuterAlt(localctx, 4)
                self.state = 144
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

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_metaEncoding

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMetaEncoding" ):
                listener.enterMetaEncoding(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMetaEncoding" ):
                listener.exitMetaEncoding(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaEncoding" ):
                return visitor.visitMetaEncoding(self)
            else:
                return visitor.visitChildren(self)




    def metaEncoding(self):

        localctx = ftrex_ftpParser.MetaEncodingContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_metaEncoding)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 147
            self.match(ftrex_ftpParser.T__5)
            self.state = 148
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MetaCmdContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_metaCmd

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMetaCmd" ):
                listener.enterMetaCmd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMetaCmd" ):
                listener.exitMetaCmd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaCmd" ):
                return visitor.visitMetaCmd(self)
            else:
                return visitor.visitChildren(self)




    def metaCmd(self):

        localctx = ftrex_ftpParser.MetaCmdContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_metaCmd)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 150
            self.match(ftrex_ftpParser.T__6)
            self.state = 151
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MetaDbNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_metaDbName

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMetaDbName" ):
                listener.enterMetaDbName(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMetaDbName" ):
                listener.exitMetaDbName(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaDbName" ):
                return visitor.visitMetaDbName(self)
            else:
                return visitor.visitChildren(self)




    def metaDbName(self):

        localctx = ftrex_ftpParser.MetaDbNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_metaDbName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 153
            self.match(ftrex_ftpParser.T__7)
            self.state = 157
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,12,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 154
                    self.matchWildcard()
                self.state = 159
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,12,self._ctx)

            self.state = 160
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MetaFTitleContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOL(self):
            return self.getToken(ftrex_ftpParser.EOL, 0)

        def getRuleIndex(self):
            return ftrex_ftpParser.RULE_metaFTitle

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMetaFTitle" ):
                listener.enterMetaFTitle(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMetaFTitle" ):
                listener.exitMetaFTitle(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMetaFTitle" ):
                return visitor.visitMetaFTitle(self)
            else:
                return visitor.visitChildren(self)




    def metaFTitle(self):

        localctx = ftrex_ftpParser.MetaFTitleContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_metaFTitle)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 162
            self.match(ftrex_ftpParser.T__8)
            self.state = 166
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,13,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 163
                    self.matchWildcard()
                self.state = 168
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,13,self._ctx)

            self.state = 169
            self.match(ftrex_ftpParser.EOL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





