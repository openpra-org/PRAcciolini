## Working with ANTLR

From `<filename>.g4`, generate lexer, parser, listener `.py` files using `antlr4 -Dlanguage=Python3 <filename>.g4`.

1. Optionally, generate rule augmented transition network diagrams as GraphViz `.dot` files using the `-atn` flag.
   ```shell
   antlr4 -atn -Dlanguage=Python3 <filename>.g4
   ```

2. Optionally, specify a package name. Also, generate the visitor logic.
   ```shell
   antlr4 -listener -visitor -package <package_name> -Dlanguage=Python3 <filename>.g4
   ```