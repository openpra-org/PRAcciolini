import os
import unittest
from collections.abc import Set
from typing import Dict, Tuple, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed

from pracciolini.grammar.openpsa.xml.event_tree.initiating_event import InitiatingEventDefinition
from pracciolini.grammar.openpsa.xml.model_data.model_data import ModelData
from pracciolini.grammar.openpsa.xml.openpsa_mef import OpsaMef
from pracciolini.utils.file_ops import FileOps
from pracciolini.grammar.openpsa.validate import read_openpsa_xml, validate_openpsa_input_xml_file


def _test_nested_reserialization(key: str, xml, converted_xml) -> Tuple[bool, Optional[str]]:
    ## test-type [re-serialization]: lxml.etree.ElementTree -> OpsaMef python subclass -> lxml.etree.ElementTree
    if xml.attrib != converted_xml.attrib:
        return False, f"re-serialized xml attributes for {key} do not match"

    tags = sorted(list([element.tag for element in xml]))
    converted_tags = sorted(list([element.tag for element in converted_xml]))
    if tags != converted_tags:
        return False, f"re-serialized xml tags for {key} do not match"

    for tag_a, tag_b in zip(tags, converted_tags):
        keys_match, message = _test_nested_reserialization(key, tag_a, tag_b)
        if not keys_match:
            return False, message

    return True, None


def _test_build_model_data_from_input_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        xml_data = read_openpsa_xml(file_path)
        model_data_xml = xml_data.find("model-data")
        if model_data_xml is not None:
            model_data = ModelData.from_xml(model_data_xml)
            if not isinstance(model_data, ModelData):
                return False, "model_data is not an instance of ModelData"
            else:
                return True, None
        else:
            # No model-data element
            return True, None
    except Exception as e:
        return False, str(e)


def _test_build_opsamef_from_input_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        xml_data = read_openpsa_xml(file_path)
        opsa_mef = OpsaMef.from_xml(xml_data)
        if not isinstance(opsa_mef, OpsaMef):
            return False, "opsa_mef is not an instance of OpsaMef"
        else:
            return True, None
    except Exception as e:
        return False, str(e)


def _test_build_initiating_event_from_input_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        xml_data = read_openpsa_xml(file_path)
        initiating_events_xml = xml_data.findall("define-initiating-event")
        for initiating_event_xml in initiating_events_xml:
            initiating_event = InitiatingEventDefinition.from_xml(initiating_event_xml)
            if not isinstance(initiating_event, InitiatingEventDefinition):
                return False, f"initiating_event is not an instance of InitiatingEventDefinition in {file_path}"
            if not initiating_event.name or initiating_event.name == "":
                return False, f"initiating_event.name is invalid in {file_path}"
            ## test-type re-serialization
            # match, msg = _test_nested_reserialization(tag, tag_xml, tag_cls_instance.to_xml())
            # if not match:
            #     return False, msg
        return True, None
    except Exception as e:
        return False, str(e)


def _test_valid_input_schema_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        is_valid = validate_openpsa_input_xml_file(file_path)
        if is_valid:
            return True, None
        else:
            return False, f"File {file_path} failed schema validation"
    except Exception as e:
        return False, str(e)


def _test_invalid_input_schema_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        is_valid = validate_openpsa_input_xml_file(file_path)
        if not is_valid:
            return True, None
        else:
            return False, f"File {file_path} unexpectedly passed schema validation"
    except Exception as e:
        return False, str(e)


def _parallel_test_wrapper(cls, test_fn, files: Set[str]):
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_file = {
            executor.submit(test_fn, file_path): file_path for file_path in files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            with cls.subTest(file_path=file_path):
                try:
                    success, error = future.result()
                    if success:
                        pass  # Test passed
                    else:
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
                "generic_pwr": os.path.join(cls.base_fixtures_path, "generic-pwr-openpsa-model/models/**/*.xml"),
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


    def test_build_model_data_from_input_xml(self):
        _parallel_test_wrapper(self, _test_build_model_data_from_input_xml_helper, self.flat_fixtures["valid"])

    def test_build_opsamef_from_input_xml(self):
        _parallel_test_wrapper(self, _test_build_opsamef_from_input_xml_helper, self.flat_fixtures["valid"])

    def test_build_initiating_event_from_input_xml(self):
        _parallel_test_wrapper(self, _test_build_initiating_event_from_input_xml_helper, self.flat_fixtures["valid"])

    def test_valid_input_schema_xml(self):
        _parallel_test_wrapper(self, _test_valid_input_schema_xml_helper, self.flat_fixtures["valid"])

    def test_invalid_input_schema_xml(self):
        _parallel_test_wrapper(self, _test_invalid_input_schema_xml_helper, self.flat_fixtures["invalid"])


if __name__ == '__main__':
    unittest.main()
