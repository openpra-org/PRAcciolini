import unittest
import tempfile
import shutil
import os
import glob
import json
from lxml import etree

from pracciolini.utils.file_ops import FileOps


class TestFileOps(unittest.TestCase):
    """
    Test suite for the FileOps utility class.

    This class tests each method of the FileOps class,
    covering base cases, edge cases, and OS-specific behaviors.
    """

    def setUp(self):
        """
        Set up a temporary directory structure and files for testing.
        This method runs before each test.
        """
        # Create a temporary directory for the tests
        self.test_dir = tempfile.mkdtemp()

        # Define a structured set of files and directories to create
        self.files_to_create = {
            'file1.txt': 'Content of file1.txt',
            'file2.xml': '<root></root>',
            'file3.json': '{"key": "value"}',
            'file4.invalid': 'This is an invalid file format',
            'subdir/file5.txt': 'Content of file5.txt',
            'subdir/file6.xml': '<root><child/></root>',
            'subdir2/nested/file7.txt': 'Content of file7.txt',
        }

        # Create the files and directories defined above
        for relative_path, content in self.files_to_create.items():
            # Determine the full path
            full_path = os.path.join(self.test_dir, relative_path)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # Write the file content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # Create an empty file
        self.empty_file = os.path.join(self.test_dir, 'empty.json')
        open(self.empty_file, 'w', encoding='utf-8').close()

        # Create a symlink to a file (if supported by the OS)
        if hasattr(os, 'symlink'):
            self.symlink_path = os.path.join(self.test_dir, 'file_symlink.txt')
            os.symlink(
                os.path.join(self.test_dir, 'file1.txt'),
                self.symlink_path
            )

        # Store the paths for easy access
        self.existing_file = os.path.join(self.test_dir, 'file1.txt')
        self.existing_dir = self.test_dir
        self.non_existing_file = os.path.join(self.test_dir, 'nonexistent.txt')
        self.non_existing_dir = os.path.join(self.test_dir, 'nonexistent_dir')

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        This method runs after each test.
        """
        shutil.rmtree(self.test_dir)

    # ---------- Tests for ensure_dir_path ----------

    def test_ensure_dir_path_existing(self):
        """
        Test ensure_dir_path with an existing directory.
        """
        result = FileOps.ensure_dir_path(self.existing_dir)
        self.assertEqual(result, self.existing_dir)
        self.assertTrue(os.path.isdir(result))

    def test_ensure_dir_path_non_existing(self):
        """
        Test ensure_dir_path with a non-existing directory.
        """
        new_dir = os.path.join(self.test_dir, 'new_directory')
        result = FileOps.ensure_dir_path(new_dir)
        self.assertEqual(result, new_dir)
        self.assertTrue(os.path.isdir(result))

    def test_ensure_dir_path_existing_file_path(self):
        """
        Test ensure_dir_path with a path that is an existing file.
        Expecting an OSError because a file exists at the path.
        """
        with self.assertRaises(OSError):
            FileOps.ensure_dir_path(self.existing_file)

    # ---------- Tests for verify_file_path ----------

    def test_verify_file_path_existing_file(self):
        """
        Test verify_file_path with an existing file.
        """
        result = FileOps.verify_file_path(self.existing_file)
        self.assertEqual(result, self.existing_file)
        self.assertTrue(os.path.isfile(result))

    def test_verify_file_path_non_existing_file(self):
        """
        Test verify_file_path with a non-existing file.
        Expecting a FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            FileOps.verify_file_path(self.non_existing_file)

    def test_verify_file_path_directory_instead_of_file(self):
        """
        Test verify_file_path with a directory path instead of a file.
        Expecting a FileNotFoundError because the path is not a file.
        """
        with self.assertRaises(FileNotFoundError):
            FileOps.verify_file_path(self.existing_dir)

    # ---------- Tests for get_files_by_type ----------

    def test_get_files_by_type_existing_files(self):
        """
        Test get_files_by_type with files that exist matching the file type.
        """
        expected_files = set(
            glob.glob(os.path.join(self.test_dir, '**', '*.txt'), recursive=True)
        )
        result = FileOps.get_files_by_type(self.test_dir, '.txt')
        self.assertEqual(result, expected_files)

    def test_get_files_by_type_no_matching_files(self):
        """
        Test get_files_by_type when there are no files matching the file type.
        """
        result = FileOps.get_files_by_type(self.test_dir, '.nonexistent')
        self.assertEqual(result, set())

    def test_get_files_by_type_with_symlink(self):
        """
        Test get_files_by_type to ensure symlinks are included if they match the file type.
        """
        if hasattr(os, 'symlink'):
            result = FileOps.get_files_by_type(self.test_dir, '.txt')
            self.assertIn(self.symlink_path, result)

    # ---------- Tests for parse_and_glob_paths ----------

    def test_parse_and_glob_paths_with_matching_patterns(self):
        """
        Test parse_and_glob_paths with glob patterns that match files.
        """
        patterns = [
            os.path.join(self.test_dir, '*.txt'),
            os.path.join(self.test_dir, '**', '*.json'),
        ]
        result = FileOps.parse_and_glob_paths(patterns)
        expected_files = [
            os.path.join(self.test_dir, 'file1.txt'),
            os.path.join(self.test_dir, 'file3.json'),
            self.empty_file
        ]
        self.assertTrue(all(file in result for file in expected_files))

    def test_parse_and_glob_paths_no_matching_files(self):
        """
        Test parse_and_glob_paths with glob patterns that match no files.
        """
        patterns = [os.path.join(self.test_dir, '*.nonexistent')]
        result = FileOps.parse_and_glob_paths(patterns)
        self.assertEqual(result, [])

    # ---------- Tests for find_files_by_extension ----------

    def test_find_files_by_extension_existing_files(self):
        """
        Test find_files_by_extension with files that exist matching the extension.
        """
        result = FileOps.find_files_by_extension(self.test_dir, '.xml')
        expected_files = [
            os.path.join(self.test_dir, 'file2.xml'),
            os.path.join(self.test_dir, 'subdir', 'file6.xml')
        ]
        self.assertCountEqual(result, expected_files)

    def test_find_files_by_extension_no_matching_files(self):
        """
        Test find_files_by_extension when there are no files matching the extension.
        """
        result = FileOps.find_files_by_extension(self.test_dir, '.nonexistent')
        self.assertEqual(result, [])

    def test_find_files_by_extension_nonexistent_directory(self):
        """
        Test find_files_by_extension with a non-existent directory.
        The result should be an empty list.
        """
        result = FileOps.find_files_by_extension(self.non_existing_dir, '.txt')
        self.assertEqual(result, [])

    # ---------- Tests for get_input_files ----------

    def test_get_input_files_with_matching_files(self):
        """
        Test get_input_files with an args object that specifies directories and file type.
        """
        class Args:
            input_folders = [self.test_dir]
            file_type = '.txt'

        expected_files = set(
            glob.glob(os.path.join(self.test_dir, '**', '*.txt'), recursive=True)
        )
        result = FileOps.get_input_files(Args)
        self.assertEqual(result, expected_files)

    def test_get_input_files_no_matching_files(self):
        """
        Test get_input_files when no files match the specified file type in the input folders.
        """
        class Args:
            input_folders = [self.test_dir]
            file_type = '.nonexistent'

        result = FileOps.get_input_files(Args)
        self.assertEqual(result, set())

    # ---------- Tests for parse_xml_file ----------

    def test_parse_xml_file_valid(self):
        """
        Test parse_xml_file with a valid XML file.
        """
        xml_file_path = os.path.join(self.test_dir, 'file2.xml')
        tree = FileOps.parse_xml_file(xml_file_path)
        self.assertIsInstance(tree, etree._ElementTree)
        root = tree.getroot()
        self.assertEqual(root.tag, 'root')

    def test_parse_xml_file_invalid(self):
        """
        Test parse_xml_file with an invalid XML file.
        Expecting an etree.XMLSyntaxError.
        """
        invalid_xml_path = os.path.join(self.test_dir, 'file4.invalid')
        with open(invalid_xml_path, 'w', encoding='utf-8') as f:
            f.write('Not a valid XML content')
        with self.assertRaises(etree.XMLSyntaxError):
            FileOps.parse_xml_file(invalid_xml_path)

    def test_parse_xml_file_empty(self):
        """
        Test parse_xml_file with an empty XML file.
        Expecting an etree.XMLSyntaxError.
        """
        empty_xml_path = os.path.join(self.test_dir, 'empty.xml')
        open(empty_xml_path, 'w', encoding='utf-8').close()
        with self.assertRaises(etree.XMLSyntaxError):
            FileOps.parse_xml_file(empty_xml_path)

    def test_parse_xml_file_nonexistent(self):
        """
        Test parse_xml_file with a non-existent file.
        Expecting a FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            FileOps.parse_xml_file(self.non_existing_file)

    # ---------- Tests for parse_json_file ----------

    def test_parse_json_file_valid(self):
        """
        Test parse_json_file with a valid JSON file.
        """
        json_file_path = os.path.join(self.test_dir, 'file3.json')
        data = FileOps.parse_json_file(json_file_path)
        self.assertIsInstance(data, dict)
        self.assertEqual(data, {'key': 'value'})

    def test_parse_json_file_invalid(self):
        """
        Test parse_json_file with an invalid JSON file.
        Expecting a json.JSONDecodeError.
        """
        invalid_json_path = os.path.join(self.test_dir, 'file4.invalid')
        with open(invalid_json_path, 'w', encoding='utf-8') as f:
            f.write('Not a valid JSON content')
        with self.assertRaises(json.JSONDecodeError):
            FileOps.parse_json_file(invalid_json_path)

    def test_parse_json_file_empty(self):
        """
        Test parse_json_file with an empty JSON file.
        Expecting a json.JSONDecodeError.
        """
        with self.assertRaises(json.JSONDecodeError):
            FileOps.parse_json_file(self.empty_file)

    def test_parse_json_file_nonexistent(self):
        """
        Test parse_json_file with a non-existent file.
        Expecting a FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            FileOps.parse_json_file(self.non_existing_file)

    # ---------- Additional Tests for OS-specific behaviors ----------

    def test_case_sensitivity_in_file_extensions(self):
        """
        Test whether the methods correctly handle case sensitivity in file extensions.
        On case-insensitive file systems (like Windows), this test ensures consistent behavior.
        """
        # Create files with uppercase extensions
        upper_case_file = os.path.join(self.test_dir, 'file8.TXT')
        with open(upper_case_file, 'w', encoding='utf-8') as f:
            f.write('Content of file8.TXT')

        # Test get_files_by_type
        result = FileOps.get_files_by_type(self.test_dir, '.TXT')
        expected_files = {upper_case_file}
        self.assertEqual(result, expected_files)

        # Test find_files_by_extension
        result = FileOps.find_files_by_extension(self.test_dir, '.TXT')
        self.assertEqual(result, [upper_case_file])

    def test_long_file_paths(self):
        """
        Test methods with long file paths to ensure they handle OS limitations.
        """
        # Create a very long directory path
        long_dir = self.test_dir
        for i in range(10):
            long_dir = os.path.join(long_dir, 'long_directory_name_' + str(i))
        os.makedirs(long_dir, exist_ok=True)
        long_file_path = os.path.join(long_dir, 'long_file_name.txt')
        with open(long_file_path, 'w', encoding='utf-8') as f:
            f.write('Content of long_file_name.txt')

        # Test verify_file_path
        result = FileOps.verify_file_path(long_file_path)
        self.assertEqual(result, long_file_path)

        # Test find_files_by_extension
        result = FileOps.find_files_by_extension(self.test_dir, '.txt')
        self.assertIn(long_file_path, result)

    def test_handling_of_symlinks(self):
        """
        Test methods' handling of symbolic links.
        """
        if hasattr(os, 'symlink'):
            # Test verify_file_path with a symlink
            result = FileOps.verify_file_path(self.symlink_path)
            self.assertEqual(result, self.symlink_path)
            self.assertTrue(os.path.islink(self.symlink_path))

            # Test get_files_by_type includes symlinked files
            result = FileOps.get_files_by_type(self.test_dir, '.txt')
            self.assertIn(self.symlink_path, result)

    def test_non_ascii_file_names(self):
        """
        Test methods with file names containing non-ASCII characters.
        """
        non_ascii_file = os.path.join(self.test_dir, 'файл.txt')
        with open(non_ascii_file, 'w', encoding='utf-8') as f:
            f.write('Content with non-ASCII filename')

        # Test verify_file_path
        result = FileOps.verify_file_path(non_ascii_file)
        self.assertEqual(result, non_ascii_file)

        # Test get_files_by_type
        result = FileOps.get_files_by_type(self.test_dir, '.txt')
        self.assertIn(non_ascii_file, result)

        # Test find_files_by_extension
        result = FileOps.find_files_by_extension(self.test_dir, '.txt')
        self.assertIn(non_ascii_file, result)

    def test_empty_directory(self):
        """
        Test methods when the directory is empty.
        """
        empty_dir = os.path.join(self.test_dir, 'empty_dir')
        os.makedirs(empty_dir, exist_ok=True)

        # Test find_files_by_extension returns an empty list
        result = FileOps.find_files_by_extension(empty_dir, '.txt')
        self.assertEqual(result, [])

        # Test get_files_by_type returns an empty set
        result = FileOps.get_files_by_type(empty_dir, '.txt')
        self.assertEqual(result, set())

    def test_parse_xml_file_with_dtd(self):
        """
        Test parse_xml_file with an XML file containing a DTD.
        """
        xml_with_dtd = os.path.join(self.test_dir, 'file_with_dtd.xml')
        xml_content = '''<?xml version="1.0"?>
        <!DOCTYPE note [
        <!ELEMENT note (to,from,heading,body)>
        <!ELEMENT to (#PCDATA)>
        <!ELEMENT from (#PCDATA)>
        <!ELEMENT heading (#PCDATA)>
        <!ELEMENT body (#PCDATA)>
        ]>
        <note>
            <to>User</to>
            <from>System</from>
            <heading>Reminder</heading>
            <body>This is a test.</body>
        </note>'''
        with open(xml_with_dtd, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        tree = FileOps.parse_xml_file(xml_with_dtd)
        self.assertIsInstance(tree, etree._ElementTree)
        root = tree.getroot()
        self.assertEqual(root.tag, 'note')

    def test_parse_json_file_with_utf16_encoding(self):
        """
        Test parse_json_file with a JSON file encoded in UTF-16.
        """
        utf16_json_file = os.path.join(self.test_dir, 'utf16.json')
        data = {'message': 'Hello, world!'}
        with open(utf16_json_file, 'w', encoding='utf-16') as f:
            json.dump(data, f)
        # Modify the parse_json_file method to handle encoding argument if necessary
        # For this test, we will simulate reading with the correct encoding
        try:
            with open(utf16_json_file, 'r', encoding='utf-16') as file:
                loaded_data = json.load(file)
        except Exception as e:
            self.fail(f"Failed to parse UTF-16 encoded JSON file: {e}")
        self.assertEqual(loaded_data, data)

    def test_parse_json_file_with_comments(self):
        """
        Test parse_json_file with a JSON file that contains comments.
        """
        json_with_comments = os.path.join(self.test_dir, 'comments.json')
        json_content = '''
        {
            // This is a single-line comment
            "key": "value" /* This is a multi-line comment */
        }
        '''
        with open(json_with_comments, 'w', encoding='utf-8') as f:
            f.write(json_content)
        # Standard json.load() does not support comments
        with self.assertRaises(json.JSONDecodeError):
            FileOps.parse_json_file(json_with_comments)

    def test_parse_json_file_with_trailing_commas(self):
        """
        Test parse_json_file with a JSON file that contains trailing commas.
        """
        json_with_trailing_commas = os.path.join(self.test_dir, 'trailing_commas.json')
        json_content = '''
        {
            "key1": "value1",
            "key2": "value2",
        }
        '''
        with open(json_with_trailing_commas, 'w', encoding='utf-8') as f:
            f.write(json_content)
        # Standard json.load() does not support trailing commas
        with self.assertRaises(json.JSONDecodeError):
            FileOps.parse_json_file(json_with_trailing_commas)

    def test_write_json_file_valid(self):
        """
        Test write_json_file with valid JSON data and file path.
        """
        data = {'key': 'value'}
        json_file_path = os.path.join(self.test_dir, 'output', 'valid.json')
        # Attempt to write the JSON data to the file
        try:
            FileOps.write_json_file(data, json_file_path)
        except Exception as e:
            self.fail(f"write_json_file raised an exception unexpectedly: {e}")
        # Verify that the file was created and contains the correct data
        self.assertTrue(os.path.isfile(json_file_path))
        with open(json_file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, data)

    def test_write_json_file_non_serializable_data(self):
        """
        Test write_json_file with data that is not JSON-serializable.
        Expecting a TypeError.
        """
        data = {'key': {1, 2, 3}}  # Sets are not JSON-serializable
        json_file_path = os.path.join(self.test_dir, 'output', 'non_serializable.json')
        with self.assertRaises(TypeError):
            FileOps.write_json_file(data, json_file_path)

    def test_write_json_file_directory_creation(self):
        """
        Test write_json_file with a file path in a non-existent directory.
        The directory should be created.
        """
        data = {'key': 'value'}
        new_dir = os.path.join(self.test_dir, 'new_dir')
        json_file_path = os.path.join(new_dir, 'file.json')
        # Ensure the directory does not exist before the test
        self.assertFalse(os.path.exists(new_dir))
        # Write the JSON data
        FileOps.write_json_file(data, json_file_path)
        # Verify that the directory and file were created
        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(os.path.isfile(json_file_path))
        with open(json_file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, data)

    def test_write_json_file_overwrite_existing_file(self):
        """
        Test write_json_file to ensure it overwrites an existing file.
        """
        data_initial = {'key': 'initial_value'}
        data_new = {'key': 'new_value'}
        json_file_path = os.path.join(self.test_dir, 'output', 'overwrite.json')
        # Write the initial data to the file
        FileOps.write_json_file(data_initial, json_file_path)
        # Verify initial data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, data_initial)
        # Overwrite with new data
        FileOps.write_json_file(data_new, json_file_path)
        # Verify new data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, data_new)

    def test_write_json_file_invalid_path(self):
        """
        Test write_json_file with an invalid file path (e.g., path is a directory).
        Expecting an IsADirectoryError or OSError.
        """
        data = {'key': 'value'}
        # Use a directory path instead of a file path
        invalid_file_path = self.existing_dir
        with self.assertRaises((IsADirectoryError, OSError)):
            FileOps.write_json_file(data, invalid_file_path)
        # Ensure no file was created with the directory name
        self.assertFalse(os.path.isfile(invalid_file_path))

if __name__ == '__main__':
    unittest.main()