import json
from typing import Any, Dict


class SysGate:
    """
    Represents a system gate.

    Attributes:
        name (str): Name of the gate.
        id (int): Identifier of the gate.
        gateid (int): Gate ID.
        gateorig (Any): Original gate reference.
        gatepos (Any): Position of the gate.
        eventid (int): Associated event ID.
        gatecomp (Any): Gate component.
        comppos (Any): Component position.
        compflag (Any): Component flag.
        gateflag (Any): Gate flag.
        gatet (Any): Gate type.
        bddsuccess (bool): Flag indicating BDD success.
        done (bool): Flag indicating completion.
    """

    def __init__(
        self,
        name: str,
        id: int,
        gateid: int,
        gateorig: Any,
        gatepos: Any,
        eventid: int,
        gatecomp: Any,
        comppos: Any,
        compflag: Any,
        gateflag: Any,
        gatet: Any,
        bddsuccess: bool,
        done: bool
    ) -> None:
        """
        Initializes a SysGate instance.

        Args:
            name (str): Name of the gate.
            id (int): Identifier of the gate.
            gateid (int): Gate ID.
            gateorig (Any): Original gate reference.
            gatepos (Any): Position of the gate.
            eventid (int): Associated event ID.
            gatecomp (Any): Gate component.
            comppos (Any): Component position.
            compflag (Any): Component flag.
            gateflag (Any): Gate flag.
            gatet (Any): Gate type.
            bddsuccess (bool): Flag indicating BDD success.
            done (bool): Flag indicating completion.
        """
        self.name: str = name
        self.id: int = id
        self.gateid: int = gateid
        self.gateorig: Any = gateorig
        self.gatepos: Any = gatepos
        self.eventid: int = eventid
        self.gatecomp: Any = gatecomp
        self.comppos: Any = comppos
        self.compflag: Any = compflag
        self.gateflag: Any = gateflag
        self.gatet: Any = gatet
        self.bddsuccess: bool = bddsuccess
        self.done: bool = done

    def to_json(self) -> json | Dict[str, Any]:
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
        return {key: val for key, val in self.__dict__.items() if val is not None}
