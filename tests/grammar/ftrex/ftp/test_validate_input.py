import os
import unittest

from pracciolini.utils.file_ops import FileOps
from pracciolini.grammar.ftrex.validate import validate_ftp_file


class TestValidateInputFtrexFtp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base path for fixtures
        cls.fixtures_path = os.path.join(cls.current_dir, '../../../fixtures/ftrex/ftp')

    def test_valid_input(self):
        """
        Test that provided ftrex FTP files pass the ANTLRv4 schema validation.
        """
        valid_schema_path = os.path.join(self.fixtures_path, 'valid')
        valid_files = FileOps.find_files_by_extension(valid_schema_path, '.ftp')
        for file_path in valid_files:
            with self.subTest(file_path=file_path):
                self.assertTrue(validate_ftp_file(file_path),
                                f"File {file_path} should be valid.")


if __name__ == '__main__':
    unittest.main()
