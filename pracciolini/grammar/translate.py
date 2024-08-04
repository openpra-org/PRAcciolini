import lxml.etree

from pracciolini.utils.file_ops import FileOps
from pracciolini.core.decorators import translation


def read_file(file_path: str) -> str:
    try:
        with open(FileOps.verify_file_path(file_path), 'r') as file:
            return file
    except Exception as e:
        print(f"An error occurred while reading the file {file_path}: {e}")
        raise


@translation('ftrex-ftp', 'openpsa-xml-ft')
def my_naive_ftrex_to_openpsa_xml(_input: str) -> lxml.etree.ElementTree:
    return ""

