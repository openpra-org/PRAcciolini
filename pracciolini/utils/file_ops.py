import glob
import os
import json

import lxml
from lxml import etree
from typing import Set, List, Any


class FileOps(object):
    """
    A utility class for performing common file operations such as verifying paths,
    searching for files by type, and parsing files in various formats.
    """

    @staticmethod
    def ensure_dir_path(path: str) -> str:
        """
        Ensures that a directory exists at the specified path.
        If the directory does not exist, it is created along with any necessary parent directories.

        Parameters:
            path (str): The path to the directory to verify or create.

        Returns:
            str: The absolute path to the verified or newly created directory.

        Raises:
            OSError: If the directory cannot be created due to permission issues or other OS-related errors.

        Example:
            >>> FileOps.ensure_dir_path('/path/to/directory')
            '/path/to/directory'
        """
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def verify_file_path(path: str) -> str:
        """
        Verifies that the given path exists and is a regular file.

        Parameters:
            path (str): The path to the file to verify.

        Returns:
            str: The absolute path to the verified file.

        Raises:
            FileNotFoundError: If the file does not exist or is not a regular file.
        """
        if os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError(f"The file {path} does not exist or is not a regular file.")

    @staticmethod
    def get_files_by_type(directory: str, file_type: str) -> Set[str]:
        """
        Searches recursively within the specified directory for files
        that match the given file extension.

        Parameters:
            directory (str): The root directory to begin the search.
            file_type (str): The file extension to filter by (e.g., '.txt', '.xml').

        Returns:
            Set[str]: A set of absolute file paths matching the specified file extension.

        Notes:
            - The search is performed recursively in all subdirectories.
            - The file_type should include the dot, e.g., '.txt'.

        Example:
            >>> FileOps.get_files_by_type('/path/to/search', '.txt')
            {'/path/to/search/file1.txt', '/path/to/search/subdir/file2.txt'}
        """
        pattern = f"{directory}/**/*{file_type}"
        return set(glob.glob(pattern, recursive=True))

    @staticmethod
    def parse_and_glob_paths(paths: List[str]) -> List[str]:
        """
        Processes a list of file path patterns, expanding any glob patterns into
        a complete list of matching file paths.

        Parameters:
            paths (List[str]): A list of file path patterns, which may include glob wildcards.

        Returns:
            List[str]: A list of file paths that match the given patterns.

        Notes:
            - Each pattern in the input list is processed with glob.glob().
            - The search is recursive, so '**' can be used to match files in subdirectories.
            - Duplicate file paths are not filtered out; use set() if unique paths are required.

        Example:
            >>> FileOps.parse_and_glob_paths(['*.txt', 'data/**/*.csv'])
            ['file1.txt', 'data/file2.csv', 'data/subdir/file3.csv']
        """
        expanded_paths = []
        for path_pattern in paths:
            expanded_paths.extend(glob.glob(path_pattern, recursive=True))
        return expanded_paths

    @staticmethod
    def find_files_by_extension(root_dir: str, extension: str) -> List[str]:
        """
        Recursively searches the specified directory and its subdirectories
        for files that have the given file extension.

        Parameters:
            root_dir (str): The root directory from which to start the search.
            extension (str): The file extension to search for (e.g., '.xml'). Include the dot.

        Returns:
            List[str]: A list of absolute file paths matching the specified extension.

        Notes:
            - The search is case-sensitive; extensions must match exactly.
            - Hidden files and directories (those starting with a dot) are included in the search.
            - The method uses os.walk to traverse the directory tree.

        Raises:
            OSError: If the root_dir is not accessible.

        Example:
            >>> FileOps.find_files_by_extension('/path/to/search', '.txt')
            ['/path/to/search/file1.txt', '/path/to/search/subdir/file2.txt']
        """
        matching_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(extension):
                    full_path = os.path.join(dirpath, filename)
                    matching_files.append(full_path)
        return matching_files

    @classmethod
    def get_input_files(cls, args) -> Set[str]:
        """
        Collects all files from the specified input folders that match the given file type.

        Parameters:
            args: An object with the following attributes:
                - input_folders (List[str]): A list of directory paths to search in.
                - file_type (str): The file extension to filter files by (e.g., '.txt').

        Returns:
            Set[str]: A set of absolute file paths matching the specified file type from all input folders.

        Notes:
            - This method aggregates files from multiple directories.
            - Duplicate file paths are automatically filtered out due to the use of a set.

        Raises:
            AttributeError: If 'args' does not have the required attributes.

        Example:
            >>> class Args:
            ...     input_folders = ['/path/to/folder1', '/path/to/folder2']
            ...     file_type = '.txt'
            >>> FileOps.get_input_files(Args)
            {'/path/to/folder1/file1.txt', '/path/to/folder2/file2.txt'}
        """
        files = set()
        for folder in args.input_folders:
            files.update(cls.get_files_by_type(folder, args.file_type))
        return files

    @classmethod
    def parse_xml_file(cls, file_path: str, remove_comments: bool = True) -> etree.ElementTree:
        """
        Parses an XML file and returns its document tree.

        Parameters:
            file_path (str): The path to the XML file to parse.
            remove_comments (bool): Whether to remove the <comment> tags.

        Returns:
            etree.ElementTree: The parsed XML document tree.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            etree.XMLSyntaxError: If there is an error parsing the XML file.
            Exception: For other exceptions that may occur during file reading or parsing.

        Notes:
            - The file is opened in binary mode to ensure proper parsing of XML files with various encodings.
            - It is recommended to handle exceptions when calling this method to catch parsing errors.

        Example:
            >>> xml_tree = FileOps.parse_xml_file('/path/to/file.xml')
            >>> root = xml_tree.getroot()
        """
        try:
            with open(cls.verify_file_path(file_path), 'rb') as file:
                return etree.parse(file, parser=lxml.etree.XMLParser(remove_comments=remove_comments))
        except etree.XMLSyntaxError as e:
            print(f"Error parsing {file_path}: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while reading the file {file_path}: {e}")
            raise

    @classmethod
    def parse_json_file(cls, file_path: str) -> Any:
        """
        Parses a JSON file and returns the data as a Python object.

        Parameters:
            file_path (str): The path to the JSON file to parse.

        Returns:
            Any: The data parsed from the JSON file, typically a dict or list.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            json.JSONDecodeError: If there is an error decoding the JSON file.
            Exception: For other exceptions that may occur during file reading.

        Notes:
            - The file is opened in text mode with UTF-8 encoding by default.
            - It is recommended to handle exceptions when calling this method to catch parsing errors.

        Example:
            >>> data = FileOps.parse_json_file('/path/to/file.json')
            >>> print(data)
            {'key': 'value'}
        """
        try:
            # Verify that the file exists and is a regular file
            verified_path = cls.verify_file_path(file_path)

            # Open the file in read mode with UTF-8 encoding
            with open(verified_path, 'r', encoding='utf-8') as file:
                # Load and return the JSON data
                return json.load(file)
        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            print(f"Error parsing JSON file {file_path}: {e}")
            raise
        except Exception as e:
            # Handle other exceptions that may occur
            print(f"An error occurred while reading the JSON file {file_path}: {e}")
            raise

    @classmethod
    def write_json_file(cls, data: Any, file_path: str) -> None:
        """
        Writes the given JSON-serializable data to the specified file path.

        Parameters:
            data (Any): The JSON-serializable data to write to file.
            file_path (str): The path where the JSON file will be written.

        Returns:
            None

        Raises:
            OSError: If the file cannot be written due to OS-related errors.
            TypeError: If 'data' is not JSON-serializable.

        Notes:
            - The method ensures that the directory of the file path exists.
            - It overwrites any existing file at the specified path.

        Example:
            >>> data = {'key': 'value'}
            >>> FileOps.write_json_file(data, '/path/to/file.json')
        """
        try:
            # Ensure the directory exists
            dir_path = os.path.dirname(file_path)
            cls.ensure_dir_path(dir_path)

            # Open the file in write mode with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as file:
                # Write the JSON data to the file
                json.dump(data, file)
        except TypeError as e:
            # Handle the case where data is not JSON-serializable
            print(f"Error writing JSON file {file_path}: {e}")
            raise
        except Exception as e:
            # Handle other exceptions that may occur
            print(f"An error occurred while writing the JSON file {file_path}: {e}")
            raise