import json
from _typeshed import SupportsWrite
from typing import Any, Dict, Optional, cast

from pracciolini.grammar.saphsolve.jsinp.jsinp import JSInp


class JSONDumper:
    """
    A utility class for converting a JSInp object into a JSON file.

    This class handles the serialization of complex nested structures within the
    JSInp, ensuring that null values are appropriately handled and
    that the output JSON is properly formatted.

    Attributes:
        quantification_input (JSInp): The JSInp instance to serialize.
    """

    def __init__(self, quantification_input: JSInp) -> None:
        """
        Initialize the JSONDumper with the provided JSInp.

        Args:
            quantification_input (JSInp): The JSInp instance containing
                the data to be serialized into JSON format.
        """
        self.quantification_input: JSInp = quantification_input

    def dump_to_json(self, file_path: str) -> None:
        """
        Serialize the JSInp object and dump it into a JSON file.

        This method converts the JSInp object into a nested dictionary
        structure suitable for JSON serialization, handling any null values, and writes
        it to the specified file path with proper indentation.

        Args:
            file_path (str): The path to the file where the JSON output will be saved.

        Raises:
            OSError: If the file cannot be opened for writing.
            TypeError: If the JSInp contains unserializable data.
        """
        # Convert JSInp object to a dictionary
        quantification_input_dict: Dict[str, Any] = {
            'version': self._handle_null_values(self.quantification_input.version),
            'saphiresolveinput': {
                'header': self._dump_header(self.quantification_input.saphiresolveinput['header']),
                'sysgatelist': [
                    self._dump_sysgate(gate)
                    for gate in self.quantification_input.saphiresolveinput.get('sysgatelist', [])
                    if gate is not None
                ],
                'faulttreelist': [
                    self._dump_faulttree(tree)
                    for tree in self.quantification_input.saphiresolveinput.get('faulttreelist', [])
                    if tree is not None
                ],
                # Include 'sequencelist' only if it exists
                **({
                    'sequencelist': [
                        self._dump_sequence(sequence)
                        for sequence in self.quantification_input.saphiresolveinput['sequencelist']
                    ]
                } if 'sequencelist' in self.quantification_input.saphiresolveinput else {}),
                'eventlist': [
                    self._dump_event(event)
                    for event in self.quantification_input.saphiresolveinput.get('eventlist', [])
                    if event is not None
                ]
            }
        }

        # Dump the dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(quantification_input_dict, cast(SupportsWrite[str], json_file), indent=4)

    def _handle_null_values(self, value: Any) -> Optional[Any]:
        """
        Handle null (None) values for JSON serialization.

        This method checks if the provided value is None and returns None explicitly.
        This is useful for ensuring that null values are represented as null in the
        resulting JSON output.

        Args:
            value (Any): The value to check for None.

        Returns:
            Optional[Any]: The original value if it's not None, otherwise None.
        """
        if value is None:
            return None
        return value

    def _dump_header(self, header: Any) -> Dict[str, Any]:
        """
        Serialize the header object of the QuantificationInput into a dictionary.

        The header contains important metadata and configuration parameters for the
        quantification process. This method converts the header and its nested objects
        into a dictionary suitable for JSON serialization.

        The header includes:
            - projectpath: The file path of the project.
            - eventtree: An object representing the event tree structure.
            - flagnum: A numerical flag indicating specific settings.
            - ftcount: The count of fault trees.
            - fthigh: The highest fault tree identifier.
            - sqcount: The count of sequences.
            - sqhigh: The highest sequence identifier.
            - becount: The count of basic events.
            - behigh: The highest basic event identifier.
            - mthigh: The highest module identifier (if applicable).
            - phhigh: The highest phase identifier (if applicable).
            - truncparam: Truncation parameters for the quantification.
            - workspacepair: A pair of workspaces used in the quantification process.
            - iworkspacepair: An initial workspace pair.

        Args:
            header (Any): The header object from the QuantificationInput.

        Returns:
            Dict[str, Any]: A dictionary representing the serialized header, with nested
            objects converted into dictionaries and null values properly handled.
        """
        header_dict: Dict[str, Any] = {
            'projectpath': header.projectpath,
            'eventtree': header.eventtree.__dict__,
            'flagnum': header.flagnum,
            'ftcount': header.ftcount,
            'fthigh': header.fthigh,
            'sqcount': header.sqcount,
            'sqhigh': header.sqhigh,
            'becount': header.becount,
            'behigh': header.behigh,
            'mthigh': header.mthigh,
            'phhigh': header.phhigh,
            'truncparam': header.truncparam.__dict__,
            'workspacepair': header.workspacepair.__dict__,
            'iworkspacepair': header.iworkspacepair.__dict__
        }
        return {key: self._handle_null_values(val) for key, val in header_dict.items() if val is not None}

    def _dump_sysgate(self, gate: Any) -> Dict[str, Any]:
        """
        Serialize a sysgate object into a dictionary for JSON output.

        System gates represent logical gates in the system model. This method
        converts the sysgate object's attributes into a dictionary, handling null
        values appropriately.

        Args:
            gate (Any): The sysgate object to serialize.

        Returns:
            Dict[str, Any]: The serialized sysgate as a dictionary, with null values handled.
        """
        return {key: self._handle_null_values(val) for key, val in gate.__dict__.items() if val is not None}

    def _dump_faulttree(self, tree: Any) -> Dict[str, Any]:
        """
        Serialize a fault tree object into a dictionary for JSON output.

        Fault trees are critical components in reliability engineering, representing
        the logical relationships leading to system failures. This method serializes
        the fault tree's header and its list of gates, ensuring that all relevant
        data is prepared for JSON serialization.

        Args:
            tree (Any): The fault tree object to serialize.

        Returns:
            Dict[str, Any]: The serialized fault tree as a dictionary, including the
            fault tree header and optionally the list of gates if available.
        """
        fault_tree_dict: Dict[str, Any] = {
            'ftheader': self._handle_null_values(tree.ftheader),
            'gatelist': None  # Default in case tree.gatelist is None
        }

        if tree.gatelist is not None:
            fault_tree_dict['gatelist'] = [
                # Serialize each gate in the gatelist
                {key: self._handle_null_values(val) for key, val in gate.__dict__.items() if val is not None}
                for gate in tree.gatelist
            ]

        return {key: self._handle_null_values(val) for key, val in fault_tree_dict.items() if val is not None}

    def _dump_sequence(self, sequence: Any) -> Dict[str, Any]:
        """
        Serialize a sequence object into a dictionary for JSON output.

        Sequences represent a series of events or steps in the system analysis.
        This method converts the sequence object's attributes into a dictionary.

        Args:
            sequence (Any): The sequence object to serialize.

        Returns:
            Dict[str, Any]: The serialized sequence as a dictionary, with null values handled.
        """
        return {key: self._handle_null_values(val) for key, val in sequence.__dict__.items() if val is not None}

    def _dump_event(self, event: Any) -> Dict[str, Any]:
        """
        Serialize an event object into a dictionary for JSON output.

        Events are fundamental elements that can represent basic events, intermediate
        events, or end events in the system model. This method serializes the event
        object's attributes into a dictionary.

        Args:
            event (Any): The event object to serialize.

        Returns:
            Dict[str, Any]: The serialized event as a dictionary, with null values handled.
        """
        return {key: self._handle_null_values(val) for key, val in event.__dict__.items() if val is not None}
