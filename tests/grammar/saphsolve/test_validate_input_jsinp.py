# import os
# import unittest
#
# from pracciolini.utils.file_ops import FileOps
# from pracciolini.grammar.saphsolve.validate import validate_openpsa_input_xml_file, validate_openpsa_project_xml_file
#
#
# class TestValidateOpenPSAInputXML(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         # Get the directory of the current script
#         cls.current_dir = os.path.dirname(os.path.abspath(__file__))
#         # Define the base path for fixtures
#         cls.fixtures_path = os.path.join(cls.current_dir, '../../../fixtures/openpsa/xml')
#
#     def test_valid_input_xml(self):
#         """
#         Test that valid XML files pass the schema validation.
#         """
#         valid_schema_path = os.path.join(self.fixtures_path, 'opsa-mef/valid')
#         valid_files = FileOps.find_files_by_extension(valid_schema_path, '.xml')
#         for file_path in valid_files:
#             with self.subTest(file_path=file_path):
#                 self.assertTrue(validate_openpsa_input_xml_file(file_path),
#                                 f"File {file_path} should be valid.")
#
#     def test_invalid_input_xml(self):
#         """
#         Test that invalid XML files fail the schema validation.
#         """
#         invalid_schema_path = os.path.join(self.fixtures_path, 'opsa-mef/invalid')
#         invalid_files = FileOps.find_files_by_extension(invalid_schema_path, '.xml')
#         for file_path in invalid_files:
#             with self.subTest(file_path=file_path):
#                 self.assertFalse(validate_openpsa_input_xml_file(file_path),
#                                  f"File {file_path} should be invalid.")
#
#     def test_valid_project_xml(self):
#         """
#         Test that valid XML files pass the schema validation.
#         """
#         valid_schema_path = os.path.join(self.fixtures_path, 'scram-project/valid')
#         valid_files = FileOps.find_files_by_extension(valid_schema_path, '.xml')
#         for file_path in valid_files:
#             with self.subTest(file_path=file_path):
#                 self.assertTrue(validate_openpsa_project_xml_file(file_path),
#                                 f"File {file_path} should be valid.")
#
#     def test_invalid_project_xml(self):
#         """
#         Test that valid XML files pass the schema validation.
#         """
#         invalid_schema_path = os.path.join(self.fixtures_path, 'scram-project/invalid')
#         invalid_files = FileOps.find_files_by_extension(invalid_schema_path, '.xml')
#         for file_path in invalid_files:
#             with self.subTest(file_path=file_path):
#                 self.assertFalse(validate_openpsa_project_xml_file(file_path), f"File {file_path} should be invalid.")
#
#
# if __name__ == '__main__':
#     unittest.main()
