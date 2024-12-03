import os
from typing import Dict, Tuple, Optional, Set
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed

import lxml
from lxml.etree import Element

from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.utils.file_ops import FileOps
from pracciolini.grammar.openpsa.validate import read_openpsa_xml, validate_openpsa_input_xml_file


def _deep_compare_xml(xml_a: Element, xml_b: Element) -> bool:

    if xml_a is None and xml_b is None:
        return True

    if xml_a is None or xml_b is None:
        raise ValueError("cannot compare with NoneType")

    if not hasattr(xml_a, "tag") or not hasattr(xml_b, "tag"):
        raise ValueError("cannot compare when elements don't have tags")

    #print(f"comparing tags {xml_a.tag} and {xml_b.tag}")

    if getattr(xml_a, "tag") != getattr(xml_b, "tag"):
        raise ValueError(f"re-serialized xml tags do not match: {xml_a.tag} != {xml_b.tag}")

    if hasattr(xml_a, "attrib") ^ hasattr(xml_b, "attrib"):
        raise ValueError("cannot compare when one element does not have attrib")

    if xml_a.attrib != xml_b.attrib:
        raise ValueError(f"attributes [{xml_a.attrib}] do not match [{xml_b.attrib}]")

    #print(f"comparing [{xml_a.tag}:{xml_a.attrib}] and [{xml_b.tag}:{xml_b.attrib}]")

    child_tags_a = sorted(element.tag for element in xml_a)
    child_tags_b = sorted(element.tag for element in xml_b)
    if child_tags_a != child_tags_b:
        raise ValueError(f"child tags [{child_tags_a}] do not match [{child_tags_b}], extra: [{set(child_tags_a) - set(child_tags_b)}]")

    children_a = (element for element in xml_a)
    children_b = (element for element in xml_b)

    all_match_so_far = True
    for child_a, child_b in zip(children_a, children_b):
        this_one_matches = _deep_compare_xml(child_a, child_b)
        all_match_so_far = all_match_so_far and this_one_matches

    return all_match_so_far

def _test_build_from_input_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    tags: Set[str] = {
        # "//opsa-mef",
        # "//define-event-tree",
        # "//define-initiating-event",
        # "//define-functional-event",
        # "//define-sequence",
        "//model-data",
        "//define-fault-tree"
    }
    xquery = "|".join(tags)
    all_match = True
    try:
        xml_data = read_openpsa_xml(file_path)
        events_xml = xml_data.xpath(xquery)
        for event_xml in events_xml:
            try:
                event = OpsaMefXmlRegistry.instance().build(event_xml)
                converted_xml = event.to_xml()
                all_match = all_match and _deep_compare_xml(event_xml, converted_xml)
            except ValueError as ve:
                return False, f"{ve}: \n {lxml.etree.tostring(event_xml)}, {lxml.etree.tostring(converted_xml)}"
    except Exception as e:
        return False, str(e)
    return True, None

def _test_valid_input_schema_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        is_valid = validate_openpsa_input_xml_file(file_path)
        if not is_valid:
            return False, f"File {file_path} failed schema validation"
        return True, None
    except Exception as e:
        return False, str(e)


def _test_invalid_input_schema_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        is_valid = validate_openpsa_input_xml_file(file_path)
        if not is_valid:
            return True, None
        return False, f"File {file_path} unexpectedly passed schema validation"
    except Exception as e:
        return False, str(e)


def _parallel_test_wrapper(cls, test_fn, files: Set[str]):
    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(test_fn, file_path): file_path for file_path in files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            with cls.subTest(file_path=file_path):
                try:
                    success, error = future.result()
                    if not success:
                        cls.fail(f"Test failed for {file_path}: {error}")
                except Exception as e:
                    cls.fail(f"Test raised exception for {file_path}: {str(e)}")


class TestBuildOpenPSAMefInputXML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base path for fixtures
        cls.base_fixtures_path = os.path.normpath(
            os.path.join(cls.current_dir, '../../../fixtures/openpsa/xml/opsa-mef')
        )
        cls.fixture_directories: Dict[str, Dict[str, str]] = {
            "valid": {
                "demo": os.path.join(cls.base_fixtures_path, "demo/*.xml"),
                "fragments": os.path.join(cls.base_fixtures_path, "valid/**/*.xml"),
                "generic": os.path.join(cls.base_fixtures_path, "generic-openpsa-models/models/**/*.xml"),
                "generic-pwr": os.path.join(cls.base_fixtures_path, "generic-pwr-openpsa-model/models/**/*.xml"),
                "synthetic": os.path.join(cls.base_fixtures_path, "synthetic-openpsa-models/models/**/ft_*.xml")
            },
            "invalid": {
                "fragments": os.path.join(cls.base_fixtures_path, "invalid/**/*.xml"),
            }
        }
        cls.flat_fixtures: Dict[str, Set[str]] = dict()

        for key_valid, dict_valid in cls.fixture_directories.items():
            if key_valid not in cls.flat_fixtures:
                cls.flat_fixtures[key_valid]: Set[str] = set()

            for key, glob_path in dict_valid.items():
                file_paths: Set[str] = set(FileOps.parse_and_glob_paths([glob_path]))
                flat_key: str = key_valid + "-" + key
                cls.flat_fixtures[flat_key] = file_paths
                (cls.flat_fixtures[key_valid]).update(file_paths)
                print(f"Total files [{flat_key}]: {len(cls.flat_fixtures[flat_key])}")

            print(f"Total files [{key_valid}]: {len(cls.flat_fixtures[key_valid])}")

        cls.flat_fixtures["benchmarks"] = cls.flat_fixtures["valid"] - cls.flat_fixtures["valid-fragments"]

    def test_build_from_input_xml(self):
        _parallel_test_wrapper(self, _test_build_from_input_xml_helper, self.flat_fixtures["benchmarks"])

    def test_valid_input_schema_xml(self):
        _parallel_test_wrapper(self, _test_valid_input_schema_xml_helper, self.flat_fixtures["valid"])

    # def test_invalid_input_schema_xml(self):
    #     _parallel_test_wrapper(self, _test_invalid_input_schema_xml_helper, self.flat_fixtures["invalid"])


if __name__ == '__main__':
    unittest.main()
