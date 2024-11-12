import os
import unittest

from pracciolini.translator.ftrex_opsamef.ftrex_opsamef import ftrex_ftp_to_opsamef_xml, ftrex_csv_to_opsamef_xml
from pracciolini.utils.file_ops import FileOps


class TestTranslateFtrexFtpToOpenPSAMefXml(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base path for fixtures
        cls.fixtures_path = os.path.join(cls.current_dir, '../fixtures/ftrex/ftp')


    def test_translate_fragments(self):
        valid_schema_path = os.path.join(self.fixtures_path, 'valid/fragment')
        ftrex_ftp_files = FileOps.find_files_by_extension(valid_schema_path, '.ftp')
        for file_path in ftrex_ftp_files:
            print(f"translating {file_path}")
            with self.subTest(file_path=file_path):
                self.assertTrue(ftrex_ftp_to_opsamef_xml(file_path),
                                f"File {file_path} should be valid.")


    def test_translate_fragments_populate_csv(self):
        # Define the base path for fixtures
        valid_schema_path = os.path.join(self.fixtures_path, 'valid/fragment')
        ftrex_ftp_files = FileOps.find_files_by_extension(valid_schema_path, '.ftp')

        csv_db_path = os.path.join(self.current_dir, '../fixtures/ftrex/db')
        ftrex_csv_files = FileOps.find_files_by_extension(csv_db_path, 'MHTGR.csv')
        for file_path in ftrex_ftp_files:
            print(f"translating {file_path}")
            with self.subTest(file_path=file_path):
                xml_doc = ftrex_ftp_to_opsamef_xml(file_path)
                ftrex_csv_to_opsamef_xml(csv_file_path=ftrex_csv_files[0], xml_doc=xml_doc)


    def test_translate_synthetic(self):
        valid_schema_path = os.path.join(self.fixtures_path, 'valid')
        ftrex_ftp_files = FileOps.find_files_by_extension(valid_schema_path, '.ftp')
        for file_path in ftrex_ftp_files:
            print(f"translating {file_path}")
            with self.subTest(file_path=file_path):
                self.assertTrue(ftrex_ftp_to_opsamef_xml(file_path),
                                f"File {file_path} should be valid.")

if __name__ == '__main__':
    unittest.main()
