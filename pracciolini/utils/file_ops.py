import glob
import os
from lxml import etree
from typing import Set, List


class FileOps(object):
    @staticmethod
    def ensure_dir_path(path: str) -> str:
        """
        Ensures that the directory exists; if not, it creates the directory.

        Parameters:
            path (str): The path to the directory.

        Returns:
            str: The verified or created directory path.
        """
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def verify_file_path(path: str) -> str:
        """
        Verifies that the given path exists and is a file.

        Parameters:
            path (str): The path to the file.

        Returns:
            str: The verified file path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")

    @staticmethod
    def get_files_by_type(directory: str, file_type: str) -> Set[str]:
        """
        Retrieves all files with the specified extension in the given directory.

        Parameters:
            directory (str): The directory to search in.
            file_type (str): The file extension to filter by.

        Returns:
            Set[str]: A set of file paths matching the specified file type.
        """
        pattern = f"{directory}/**/*{file_type}"
        return set(glob.glob(pattern, recursive=True))

    @staticmethod
    def parse_and_glob_paths(paths: List[str]) -> List[str]:
        """
        Expands the glob patterns in the list of paths and returns a list of actual file paths.

        Parameters:
            paths (List[str]): A list of path patterns possibly containing glob patterns.

        Returns:
            List[str]: A list of expanded file paths.
        """
        expanded_paths = []
        for path_pattern in paths:
            expanded_paths.extend(glob.glob(path_pattern, recursive=True))
        return expanded_paths

    @staticmethod
    def find_files_by_extension(root_dir: str, extension: str) -> List[str]:
        """
        Recursively finds all files with the given extension in the specified directory and its subdirectories.

        Args:
            root_dir (str): The root directory from which to start searching.
            extension (str): The file extension to search for. Include the dot, e.g., '.xml'.

        Returns:
            List[str]: A list of full file paths matching the specified extension.
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
        Collects all files from specified input folders based on the file type provided in args.

        Parameters:
            args: An object with attributes 'input_folders' (list of directory paths) and 'file_type' (str).

        Returns:
            Set[str]: A set of file paths from all specified input folders filtered by file type.
        """
        files = set()
        for folder in args.input_folders:
            files.update(cls.get_files_by_type(folder, args.file_type))
        return files

    @classmethod
    def parse_xml_file(cls, file_path: str) -> etree.ElementTree:
        """
        Parses an XML or Relax NG schema file.

        Args:
            file_path (str): Path to the XML or Relax NG schema file.

        Returns:
            etree._ElementTree: Parsed XML document tree.
        """
        try:
            with open(FileOps.verify_file_path(file_path), 'rb') as file:
                return etree.parse(file)
        except etree.XMLSyntaxError as e:
            print(f"Error parsing {file_path}: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while reading the file {file_path}: {e}")
            raise
