import pathlib
import enum
import lxml.etree as etree

from typing import Callable, Optional
from typing_extensions import Annotated

import typer

from rich.console import Console

from pracciolini.translator.ftrex_opsamef import ftrex_ftp_to_opsamef_xml

__all__ = [
    "app",
]

console = Console()
app = typer.Typer(pretty_exceptions_enable=True)


class InFileFormat(enum.Enum):
    ftp = '.ftp'


class OutFileFormat(enum.Enum):
    xml = '.xml'


allowed_translations: dict[tuple[InFileFormat, OutFileFormat], Callable] = {
    (InFileFormat.ftp, OutFileFormat.xml): ftrex_ftp_to_opsamef_xml
}



@app.command()
def translate(
        filepath: Annotated[
            pathlib.Path,
            typer.Argument(
                exists=True,
                file_okay=True,
                dir_okay=False,
                writable=False,
                readable=True,
                resolve_path=True,
                help="Path to the file to translate",
            ),
        ],
        output: Annotated[
            Optional[pathlib.Path],
            typer.Option(
                '--output', '-o',
                writable=True,
                resolve_path=True,
                help="Output file name",
                rich_help_panel="Parameters",
            ),
        ] = None,
        in_format: Annotated[
           Optional[InFileFormat], typer.Option(
               rich_help_panel="Parameters",
           )
        ] = None,
        out_format: Annotated[
            OutFileFormat, typer.Option(
                rich_help_panel="Parameters",
            )
        ] = OutFileFormat.xml,
        validate: Annotated[
            bool, typer.Option(
                rich_help_panel="Parameters",
                help="Validate the output file",
            )
        ] = True,
) -> None:
    if in_format is None:
        guess = filepath.suffix.lower()
        if guess not in InFileFormat:
            raise typer.BadParameter(f'Unknown input file format: {filepath.suffix}')
        in_format = InFileFormat(guess)

    if output is None:
        output = filepath.with_suffix(out_format.value)

    translation = (in_format, out_format)
    translator = allowed_translations.get(translation, None)
    if translator is None:
        raise typer.BadParameter(f"Translation {in_format.value} -> {out_format.value} is not supported")

    console.print(f"Translating:\n"
                  f"  {filepath} ({in_format.value})\n"
                  f"to:\n"
                  f"  {output} ({out_format.value})")

    xml = ftrex_ftp_to_opsamef_xml(filepath, validate=validate)
    xml.write(output, pretty_print=True, xml_declaration=True, encoding="utf-8")

@app.callback()
def callback():
    pass

if __name__ == "__main__":
    app()
