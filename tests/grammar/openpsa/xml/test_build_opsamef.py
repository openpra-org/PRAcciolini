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


def _test_build_model_data_from_input_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Helper function to test building a ModelData object from a given OpenPSA XML file path.

    This function attempts to read the XML data using `read_openpsa_xml`, then locates the
    "model-data" element within the parsed data. If the element exists, it attempts to
    construct a `ModelData` object from it using the `from_xml` class method.

    This function performs basic validation to ensure the constructed object is indeed an
    instance of `ModelData`. If successful, it returns `(True, None)`. Otherwise, it returns
    `(False, error_message)` where `error_message` describes the encountered issue.

    Args:
        file_path (str): Path to the OpenPSA XML file.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating success and
                                      an optional error message if unsuccessful.
    """

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
    """
    Helper function to test building an OpsaMef object from a given OpenPSA XML file path.

    This function follows a similar structure as `_test_build_model_data_from_input_xml_helper`.
    It attempts to read the XML data, construct an `OpsaMef` object using `from_xml`, and
    performs basic validation to ensure the constructed object is an instance of `OpsaMef`.

    Args:
        file_path (str): Path to the OpenPSA XML file.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating success and
                                      an optional error message if unsuccessful.
    """

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
    """
    This helper function tests the ability to build an `InitiatingEventDefinition` object
    from an OpenPSA input XML file containing a "define-initiating-event" element.

    Args:
        file_path (str): The path to the OpenPSA input XML file.

    Returns:
        Tuple[bool, Optional[str]]:
            - A tuple containing:
                - success (bool): True if the test passed, False otherwise.
                - error_message (Optional[str]): An error message if the test failed,
                                                 None otherwise.

    Raises:
        Exception: Any unexpected exception encountered during processing.

    This function iterates through all "define-initiating-event" elements in the
    provided XML file and attempts to create an `InitiatingEventDefinition` object
    from each one. It performs the following checks:

        - Whether the created object is an instance of `InitiatingEventDefinition`.
        - Whether the `name` attribute of the initiating event is valid (not empty).

    If any of these checks fail, the function returns `False` and a specific error
    message indicating the issue. Otherwise, it returns `True` and `None`.

    **Example:**

    ```python
    success, error_message = _test_build_initiating_event_from_input_xml_helper("my_openpsa_file.xml")
    if success:
        print("Successfully validated initiating events in the XML file.")
    else:
        print(f"Error: {error_message}")
    """

    try:
        xml_data = read_openpsa_xml(file_path)
        initiating_events_xml = xml_data.findall("define-initiating-event")
        for initiating_event_xml in initiating_events_xml:
            initiating_event = InitiatingEventDefinition.from_xml(initiating_event_xml)
            if not isinstance(initiating_event, InitiatingEventDefinition):
                return False, f"initiating_event is not an instance of InitiatingEventDefinition in {file_path}"
            if not initiating_event.name or initiating_event.name == "":
                return False, f"initiating_event.name is invalid in {file_path}"
            initiating_event_converted_xml = initiating_event.to_xml()
            if initiating_event.name != initiating_event_converted_xml.get("name"):
                return False, f"converted initiating_event.name did not convert back to xml"
            if initiating_event.event_tree != initiating_event_converted_xml.get("event-tree"):
                return False, f"converted initiating_event.event-tree did not convert back to xml"
        return True, None
    except Exception as e:
        return False, str(e)


def _test_valid_input_schema_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validates an OpenPSA input XML file against its schema.

    Args:
        file_path (str): The path to the OpenPSA input XML file.

    Returns:
        Tuple[bool, Optional[str]]:
            - A tuple containing:
                - success (bool): True if the file is valid, False otherwise.
                - error_message (Optional[str]): An error message if the file is invalid,
                                                  None otherwise.

    Raises:
        Exception: Any unexpected exception encountered during validation.

    This function uses the `validate_openpsa_input_xml_file` function to check if
    the provided XML file conforms to the OpenPSA schema. If the validation
    succeeds, the function returns `True` and `None`. Otherwise, it returns `False`
    and an error message indicating the validation failure.

    **Example:**

    ```python
    success, error_message = _test_valid_input_schema_xml_helper("my_openpsa_file.xml")
    if success:
        print("The XML file is valid.")
    else:
        print(f"Error: {error_message}")
    ```
    """

    try:
        is_valid = validate_openpsa_input_xml_file(file_path)
        if is_valid:
            return True, None
        else:
            return False, f"File {file_path} failed schema validation"
    except Exception as e:
        return False, str(e)


def _test_invalid_input_schema_xml_helper(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Tests if an invalid OpenPSA input XML file fails schema validation.

    Args:
        file_path (str): The path to the invalid OpenPSA input XML file.

    Returns:
        Tuple[bool, Optional[str]]:
            - A tuple containing:
                - success (bool): True if the file is invalid and fails validation,
                                  False otherwise.
                - error_message (Optional[str]): An error message if the file is valid
                                                  unexpectedly, None otherwise.

    Raises:
        Exception: Any unexpected exception encountered during validation.

    This function validates the provided XML file against the OpenPSA schema.
    If the file is invalid and the validation fails as expected, the function
    returns `True` and `None`. If the file is valid unexpectedly, it returns `False`
    and an error message indicating the unexpected validation success.

    **Example:**

    ```python
    success, error_message = _test_invalid_input_schema_xml_helper("invalid_openpsa_file.xml")
    if success:
        print("The XML file is invalid as expected.")
    else:
        print(f"Error: {error_message}")
    ```
    """

    try:
        is_valid = validate_openpsa_input_xml_file(file_path)
        if not is_valid:
            return True, None
        else:
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