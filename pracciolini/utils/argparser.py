import glob
from argparse import Namespace
import sys
from typing import Set

import argparse
import os


def parse_and_glob_paths(paths):
    """
    Expands the glob patterns in the list of paths and returns a list of actual file paths.
    """
    expanded_paths = []
    for path_pattern in paths:
        # Glob the path pattern and extend the list of expanded paths
        expanded_paths.extend(glob.glob(path_pattern))
    return expanded_paths


def parse_args(argv: None) -> Namespace:
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parse and convert text files.")

    # Add an argument for accepting a list of paths (with globbing support)
    parser.add_argument('paths', nargs='+', help='List of paths to text files, supports glob patterns.')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


class ArgParser(argparse.ArgumentParser):

    @staticmethod
    def dir_path(string):
        os.makedirs(string, exist_ok=True)
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    @staticmethod
    def file_path(string):
        if os.path.exists(string):
            return string
        else:
            raise FileNotFoundError(string)

    @staticmethod
    def is_type_of_file(file, file_type):
        ext = os.path.splitext(file)[1]
        if ext == file_type:
            return file
        else:
            return False

    @staticmethod
    def is_excel_file(file):
        return ArgParser.is_type_of_file(file, '.xlsx') or ArgParser.is_type_of_file(file, '.xls')

    @staticmethod
    def is_xml_file(file):
        return ArgParser.is_type_of_file(file, '.xml')

    @staticmethod
    def get_all_files_of_type_in_dir(rule, path) -> Set:
        files = set()
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path) and rule(full_path):
                files.add(full_path)
        return files

    @staticmethod
    def alteast_one(args) -> bool:
        for arg in args:
            if arg in sys.argv:
                return True
        return False

    def add_arguments(self):
        # For generating fault tree and flow xml files
        self.add_argument('-g', '--generate', action='store_true',
                          required=not (ArgParser.alteast_one(
                              args=['--post-process', '-p', '-q', '--quantify', '-m', '--pickle'])),
                          help='Generate fault trees')
        self.add_argument('-i', '--input-folders', type=ArgParser.dir_path, nargs="+",
                          required=ArgParser.alteast_one(args=['--generate', '-g', '-m', '--pickle']),
                          help='Directory path for supply chain input files [xls(x) or pickle]')
        self.add_argument('-o', '--output-folder', type=ArgParser.dir_path, nargs=1,
                          required=ArgParser.alteast_one(args=['--generate', '-g', '--quantify', '-q']),
                          help='Directory path for generated fault-tree/flow xml files')
        self.add_argument('-u', '--uncertainty', action='store_true', required=False,
                          help='Flag to generate uncertain basic events')
        self.add_argument('-b', "--use-backup-facilities", action='store_true', required=False,
                          help='Do not ignore backup facilities')
        self.add_argument('-id', '--ignore-dependencies', action='store_true', required=False,
                          help='Allow duplicate basic events')
        self.add_argument('-m', '--pickle', nargs=1,
                          required=not (ArgParser.alteast_one(
                              args=['--generate', '-g', '-q', '--quantify', '-p', '--post-process'])),
                          help='File path for pickled dataframe')
        self.add_argument('-t', '--type', type=str, default='FDA', required=False,
                          help='Type of supply chain to be generated')

        self.add_argument('-q', '--quantify', action='store_true',
                          required=not (ArgParser.alteast_one(
                              args=['--generate', '-g', '-m', '--pickle', '-p', '--post-process'])),
                          help='Quantify provided fault trees using SCRAM')
        self.add_argument('--num-trials', type=int, default=10000, required=False,
                          help='Number of Monte-Carlo trials to use when sampling a distribution')
        self.add_argument('--num-quantiles', type=int, default=20, required=False,
                          help='Number of quantiles to generate when sampling a distribution')
        self.add_argument('--num-bins', type=int, default=20, required=False,
                          help='Number of bins for histograms')
        self.add_argument('--importance', action='store_true', required=False, default=True,
                          help='Whether to generate importance measures or not')

        # For post-processing quantified fault tree to generate output excel sheet
        self.add_argument('-p', '--post-process', action='store_true',
                          required=not (
                              ArgParser.alteast_one(args=['--generate', '-g', '-q', '--quantify', '-m', '--pickle'])))
        self.add_argument('-r', '--report-xml-file', type=ArgParser.is_xml_file, nargs=1,
                          required=ArgParser.alteast_one(args=['-p', '--post-process', '--quantify', '-q']),
                          help='Path for report xml file')
        self.add_argument('-f', '--flow-xml-file', type=ArgParser.is_xml_file, nargs=1,
                          required=ArgParser.alteast_one(args=['--generate', '-g', '-p', '--post-process']),
                          help='Path for flow xml file')
        self.add_argument('-O', '--output-file', nargs=1, required=ArgParser.alteast_one(args=['-p', '--post-process']),
                          help='File path for merged excel sheet')
