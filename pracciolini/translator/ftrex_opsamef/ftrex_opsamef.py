from lxml.etree import ElementTree
from lxml import etree
import pandas as pd

from pracciolini.core.decorators import translation
from pracciolini.grammar.ftrex.validate import read_ftrex_ftp
from pracciolini.grammar.openpsa.validate import validate_openpsa_input_xml
from pracciolini.translator.ftrex_opsamef.opsamef_xml_visitor import OpsaMefXmlVisitor


@translation('filepath_ftrex_ftp', 'ft_opsamef_xml')
def ftrex_ftp_to_opsamef_xml(file_path: str) -> ElementTree:
    xml_doc: ElementTree = ElementTree()
    try:
        parse_tree = read_ftrex_ftp(file_path)
        visitor = OpsaMefXmlVisitor()
        xml_doc = visitor.build_xml(parse_tree)
        validate_openpsa_input_xml(xml_doc)
    except Exception as e:
        print(f"An error occurred during translation: {e}")
    print(etree.tostring(xml_doc, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode())
    return xml_doc


def ftrex_csv_to_opsamef_xml(csv_file_path: str, xml_doc: ElementTree) -> ElementTree:
    try:
        validate_openpsa_input_xml(xml_doc)
        df = pd.read_csv(csv_file_path, usecols=['NAME', 'DESC'])
        # apply
        for name, desc in zip(df['NAME'], df['DESC']):
            #
            pass
    except Exception as e:
        print(f"An error occurred during translation: {e}")
    print(etree.tostring(xml_doc, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode())
    return xml_doc

# @translation with a given source_type, and input where input_type==string can use
