from antlr4 import FileStream, CommonTokenStream, ParseTreeVisitor
from antlr4.error.ErrorListener import ErrorListener

from pracciolini.core.decorators import load
from pracciolini.grammar.ftrex.ftp.lexer import ftrex_ftpLexer
from pracciolini.grammar.ftrex.ftp.parser import ftrex_ftpParser


class FtrexFtpValidationErrorListener(ErrorListener):
    """
    Custom error listener for ANTLR4 to capture syntax errors during FTP file parsing.
    """

    def __init__(self):
        super(FtrexFtpValidationErrorListener, self).__init__()
        self.errors: list[str] = []  # List to store error messages

    def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
        """
        Overridden method to handle syntax errors.

        Args:
            recognizer: The ANTLR4 recognizer.
            offending_symbol: The offending symbol in the input stream.
            line: The line number of the error.
            column: The column number of the error.
            msg: The error message.
            e: The exception raised during parsing.
        """
        self.errors.append(f"Syntax error at line {line}, column {column}: {msg}")


@load("ftrex_ftp", ".ftp")
def read_ftrex_ftp(file_path: str) -> ParseTreeVisitor:
    """
    Reads an FTP file and returns a parse tree.

    Args:
        file_path (str): The path to the FTP file.

    Returns:
        ParseTreeVisitor: The parse tree representing the FTP file's structure.
    """

    try:
        # Create an input stream from the file
        input_stream = FileStream(file_path)

        # Create a custom error listener to capture syntax errors
        error_listener = FtrexFtpValidationErrorListener()

        # Create a lexer and add the error listener
        lexer = ftrex_ftpLexer(input_stream)
        lexer.addErrorListener(error_listener)

        # Tokenize the input stream
        stream = CommonTokenStream(lexer)

        # Create a parser and add the error listener
        parser = ftrex_ftpParser(stream)
        parser.addErrorListener(error_listener)

        # Parse the file starting from the 'file_' rule
        tree = parser.file_()

        # Check for syntax errors and print them if found
        if error_listener.errors:
            print(tree)
            print("Errors found during parsing:")
            for error in error_listener.errors:
                print(error)

        return tree

    except Exception as e:
        print(f"An error occurred during validation: {e}")


def validate_ftp_file(file_path: str) -> bool:
    """
    Validates an FTP file by attempting to parse it.

    Args:
        file_path (str): The path to the FTP file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """

    try:
        read_ftrex_ftp(file_path)
        return True
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return False
