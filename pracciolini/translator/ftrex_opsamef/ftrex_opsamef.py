from lxml.etree import ElementTree
from lxml import etree

from pracciolini.core.decorators import translation
from pracciolini.grammar.ftrex.validate import read_ftrex_ftp
from pracciolini.translator.ftrex_opsamef.opsamef_xml_visitor import OpsaMefXmlVisitor


@translation('filepath_ftrex_ftp', 'ft_opsamef_xml')
def ftrex_ftp_to_opsamef_xml(file_path: str) -> ElementTree:
    xml_doc: ElementTree = ElementTree()
    try:
        parse_tree = read_ftrex_ftp(file_path)
        visitor = OpsaMefXmlVisitor()
        xml_doc = visitor.build_xml(parse_tree)
    except Exception as e:
        print(f"An error occurred during translation: {e}")
    print(etree.tostring(xml_doc, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode())
    return xml_doc

# @translation with a given source_type, and input where input_type==string can use
