import os
import unittest
from typing import Dict, Set, Tuple, Optional

from pracciolini.translator.opsamef_expr.opsamef_expr import opsamef_xml_to_expr
from pracciolini.utils.file_ops import FileOps
from tests import _parallel_test_wrapper


class TestTranslateOpenPSAXmlToExpr(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base path for fixtures
        cls.base_fixtures_path = os.path.normpath(
            os.path.join(cls.current_dir, '../fixtures/openpsa/xml/opsa-mef')
        )
        cls.fixture_directories: Dict[str, Dict[str, str]] = {
            "valid": {
                "demo": os.path.join(cls.base_fixtures_path, "demo/*.xml"),
                "fragments": os.path.join(cls.base_fixtures_path, "valid/**/*.xml"),
                "generic": os.path.join(cls.base_fixtures_path, "generic-openpsa-models/models/**/*baobab1*.xml"),
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

        cls.flat_fixtures["valid-fragments"] -= cls.flat_fixtures["ignored"]
        cls.flat_fixtures["valid"] -= cls.flat_fixtures["ignored"]


    @staticmethod
    def _test_translate_demo(file_path: str) -> Tuple[bool, Optional[str]]:
        try:
            return opsamef_xml_to_expr(file_path), None
        except Exception as e:
            return False, str(e)

    def test_translate_demo(self):
        _parallel_test_wrapper(self, self._test_translate_demo, self.flat_fixtures["valid-generic"], max_workers=1)


if __name__ == '__main__':
    unittest.main()