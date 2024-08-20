from antlr4 import FileStream, CommonTokenStream, ParseTreeVisitor
from antlr4.error.ErrorListener import ErrorListener

from pracciolini.core.decorators import translation
from pracciolini.grammar.ftrex.ftp.lexer import ftrex_ftpLexer
from pracciolini.grammar.ftrex.ftp.parser import ftrex_ftpParser


class FtrexFtpValidationErrorListener(ErrorListener):
    def __init__(self):
        super(FtrexFtpValidationErrorListener, self).__init__()
        self.errors = []

    def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
        self.errors.append(f"Syntax error at line {line}, column {column}: {msg}")


@translation("filepath_ftrex_ftp", "ftrex_ftp")
def read_ftrex_ftp(file_path: str) -> ParseTreeVisitor:
    try:
        # Read the file using ANTLR's FileStream
        input_stream = FileStream(file_path)
        error_listener = FtrexFtpValidationErrorListener()

        # Create a lexer with the input stream
        lexer = ftrex_ftpLexer(input_stream)
        # lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)

        # Tokenize the input stream
        stream = CommonTokenStream(lexer)

        # Create a parser with the token stream
        parser = ftrex_ftpParser(stream)
        # parser.removeErrorListeners()

        parser.addErrorListener(error_listener)

        # Parse the file starting from the 'file_' rule
        tree = parser.file_()

        # Check for syntax errors
        if error_listener.errors:
            print(tree)
            print("Errors found during parsing:")
            for error in error_listener.errors:
                print(error)

        return tree
    except Exception as e:
        print(f"An error occurred during validation: {e}")


def validate_ftp_file(file_path: str) -> bool:
    try:
        read_ftrex_ftp(file_path)
        return True
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return False
