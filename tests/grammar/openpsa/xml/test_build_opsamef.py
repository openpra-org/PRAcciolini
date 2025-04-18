import os
from typing import Dict, Tuple, Optional, Set
import unittest

import lxml

from pracciolini.grammar.openpsa.opsamef import OpsaMefXmlRegistry
from pracciolini.utils.file_ops import FileOps
from pracciolini.grammar.openpsa.validate import read_openpsa_xml, validate_openpsa_input_xml_file
from pracciolini.utils.xml import deep_compare_xml
from tests import _parallel_test_wrapper


def _test_build_from_input_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    tags: Set[str] = {
        "//opsa-mef",
    }
    xquery = "|".join(tags)
    all_match = True
    try:
        xml_data = read_openpsa_xml(file_path)
        events_xml = xml_data.xpath(xquery)
        for event_xml in events_xml:
            converted_xml = None
            try:
                event = OpsaMefXmlRegistry.instance().build(event_xml)
                converted_xml = event.to_xml()
                #print(lxml.etree.tostring(event_xml), lxml.etree.tostring(converted_xml))
                all_match = all_match and deep_compare_xml(event_xml, converted_xml)
            except ValueError as ve:
                converted = lxml.etree.tostring(converted_xml) if converted_xml is not None else "FAILED"
                return False, f"{ve}: \n {lxml.etree.tostring(event_xml)}, {converted}"
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
            },
            "ignored": {
                "fragments-model": os.path.join(cls.base_fixtures_path, "valid/model/**/*.xml"),
                "fragments-eta-test-event": os.path.join(cls.base_fixtures_path, "valid/eta/test-event/**/*.xml"),
                "fragments-eta-if-then-else": os.path.join(cls.base_fixtures_path, "valid/eta/if-then-else/**/*.xml"),
                "fragments-eta-rule": os.path.join(cls.base_fixtures_path, "valid/eta/rule/**/*.xml"),
                "fragments-eta-set-event": os.path.join(cls.base_fixtures_path, "valid/eta/set-event/**/*.xml"),
                "fragments-fta-ccf": os.path.join(cls.base_fixtures_path, "valid/fta/ccf/**/*.xml"),
                "fragments-fta-complex": os.path.join(cls.base_fixtures_path, "valid/fta/complex/**/*.xml"),
                "fragments-fta-component": os.path.join(cls.base_fixtures_path, "valid/fta/component/**/*.xml"),
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
        cls.flat_fixtures["valid-fragments"] -= cls.flat_fixtures["ignored"]
        cls.flat_fixtures["valid"] -= cls.flat_fixtures["ignored"]

    def test_build_from_input_xml(self):
        _parallel_test_wrapper(self, _test_build_from_input_xml_helper, self.flat_fixtures["valid"])

    def test_valid_input_schema_xml(self):
        _parallel_test_wrapper(self, _test_valid_input_schema_xml_helper, self.flat_fixtures["valid"])

    # def test_invalid_input_schema_xml(self):
    #     _parallel_test_wrapper(self, _test_invalid_input_schema_xml_helper, self.flat_fixtures["invalid"])


if __name__ == '__main__':
    unittest.main()
