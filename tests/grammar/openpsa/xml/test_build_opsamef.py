import os
import unittest

import lxml.etree

from pracciolini.grammar.openpsa.xml.openpsa_mef import OpsaMef
from pracciolini.utils.file_ops import FileOps
from pracciolini.grammar.openpsa.validate import validate_openpsa_input_xml_file, read_openpsa_xml


class TestValidateOpenPSAInputXML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base path for fixtures
        cls.fixtures_path = os.path.join(cls.current_dir, '../../../fixtures/openpsa/xml/opsa-mef/demo')

    def test_valid_input_xml(self):
        demo_files = FileOps.find_files_by_extension(self.fixtures_path, '.xml')
        for file_path in demo_files:
            xml_data: lxml.etree.ElementTree = read_openpsa_xml(file_path)
            model = OpsaMef.from_xml(xml_data)
            print(model)
            with self.subTest(file_path=file_path):
                self.assertTrue(validate_openpsa_input_xml_file(file_path),
                                f"File {file_path} should be valid.")


if __name__ == '__main__':
    unittest.main()
