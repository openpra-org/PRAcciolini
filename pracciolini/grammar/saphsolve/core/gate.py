from typing import Optional, List, Any


class Gate:
    """
    Represents a gate within a fault tree.

    Attributes:
        gateid (int): Identifier of the gate.
        gatetype (str): Type of the gate.
        numinputs (int): Number of inputs to the gate.
        gateinput (Optional[List[Any]]): List of gate inputs.
        eventinput (Optional[List[Any]]): List of event inputs.
        compeventinput (Optional[List[Any]]): List of component event inputs.
    """

    def __init__(
        self,
        gateid: int,
        gatetype: str,
        numinputs: int,
        gateinput: Optional[List[Any]] = None,
        eventinput: Optional[List[Any]] = None,
        compeventinput: Optional[List[Any]] = None
    ) -> None:
        """
        Initializes a Gate instance.

        Args:
            gateid (int): Identifier of the gate.
            gatetype (str): Type of the gate.
            numinputs (int): Number of inputs to the gate.
            gateinput (Optional[List[Any]]): List of gate inputs.
            eventinput (Optional[List[Any]]): List of event inputs.
            compeventinput (Optional[List[Any]]): List of component event inputs.
        """
        self.gateid: int = gateid
        self.gatetype: str = gatetype
        self.numinputs: int = numinputs
        self.gateinput: Optional[List[Any]] = gateinput
        self.eventinput: Optional[List[Any]] = eventinput
        self.compeventinput: Optional[List[Any]] = compeventinput
