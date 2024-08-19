import lxml

from pracciolini.core.decorators import translation
from pracciolini.grammar.ftrex.validate import read_ftrex_ftp
from pracciolini.grammar.openpsa.xml.openpsa_mef import OpsaMef


@translation('filepath_ftrex_ftp', 'ft_opsamef_xml')
def ftrex_ftp_to_opsamef_xml(file_path: str) -> OpsaMef:
    opsa_mef = OpsaMef(name=file_path)
    try:
        parse_tree = read_ftrex_ftp(file_path)
    except Exception as e:
        print(f"An error occurred during translation: {e}")

    return opsa_mef

# @translation with a given source_type, and input where input_type==string can use
