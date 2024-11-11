import os

import lxml.etree
from lxml import etree

from pracciolini.core.decorators import translation
from pracciolini.utils.file_ops import FileOps


def validate_rng_xml(xml_doc: etree.ElementTree, schema_doc: etree.ElementTree) -> bool:
    """
    Validates an XML document against a Relax NG schema.

    Args:
        xml_doc (etree._ElementTree): The XML document tree to be validated.
        schema_doc (etree._ElementTree): The Relax NG schema document tree.

    Returns:
        bool: True if the XML is valid according to the schema, False otherwise.
    """
    try:
        relaxng_schema = etree.RelaxNG(schema_doc)

        # Validate the XML against the schema
        is_valid = relaxng_schema.validate(xml_doc)

        if not is_valid:
            for error in relaxng_schema.error_log:
                print(error.message)

        return is_valid

    except etree.RelaxNGParseError as e:
        print(f"Error parsing Relax NG schema: {e}")
        return False
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return False


def validate_rng_xml_file(xml_path: str, schema_path: str) -> bool:
    try:
        xml_doc = FileOps.parse_xml_file(xml_path)
        rng_schema = FileOps.parse_xml_file(schema_path)
    except etree.XMLSyntaxError as e:
        print(f"An error occurred during validation: {e}")
        return False
    return validate_rng_xml(xml_doc=xml_doc, schema_doc=rng_schema)


def validate_openpsa_input_xml_file(xml_path: str) -> bool:
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(current_dir, 'schema/input.rng')
    return validate_rng_xml_file(xml_path=xml_path, schema_path=schema_path)


@translation("filepath_opsamef_xml", "opsamef_xml")
def read_openpsa_xml(xml_file_path: str) -> lxml.etree.ElementTree:
    try:
        if validate_openpsa_input_xml_file(xml_file_path):
            xml_doc: lxml.etree.ElementTree = FileOps.parse_xml_file(xml_file_path)
            return xml_doc
        else:
            raise etree.XMLSyntaxError("")
    except Exception as e:
        print(f"An error occurred while reading file: {e}")


def validate_openpsa_input_xml(xml_doc: etree.ElementTree) -> bool:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, 'schema/input.rng')
        rng_schema = FileOps.parse_xml_file(schema_path)
    except etree.XMLSyntaxError as e:
        print(f"An error occurred during validation: {e}")
        return False
    return validate_rng_xml(xml_doc=xml_doc, schema_doc=rng_schema)


def validate_openpsa_report_xml_file(xml_path: str) -> bool:
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(current_dir, 'schema/output.rng')
    return validate_rng_xml_file(xml_path=xml_path, schema_path=schema_path)


def validate_openpsa_project_xml_file(xml_path: str) -> bool:
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(current_dir, 'schema/project.rng')
    return validate_rng_xml_file(xml_path=xml_path, schema_path=schema_path)
