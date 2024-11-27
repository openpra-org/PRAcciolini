from typing import Any


class Event:
    """
    Represents an event in the system.

    Attributes:
        id (int): Event identifier.
        corrgate (Any): Corresponding gate.
        name (str): Name of the event.
        evworkspacepair (Any): Workspace pair associated with the event.
        value (Any): Value associated with the event.
        initf (Any): Initialization flag.
        processf (Any): Process flag.
        calctype (str): Calculation type.
    """

    def __init__(
        self,
        id: int,
        corrgate: Any,
        name: str,
        evworkspacepair: Any,
        value: Any,
        initf: Any,
        processf: Any,
        calctype: str
    ) -> None:
        """
        Initializes an Event instance.

        Args:
            id (int): Event identifier.
            corrgate (Any): Corresponding gate.
            name (str): Name of the event.
            evworkspacepair (Any): Workspace pair associated with the event.
            value (Any): Value associated with the event.
            initf (Any): Initialization flag.
            processf (Any): Process flag.
            calctype (str): Calculation type.
        """
        self.id: int = id
        self.corrgate: Any = corrgate
        self.name: str = name
        self.evworkspacepair: Any = evworkspacepair
        self.value: Any = value
        self.initf: Any = initf
        self.processf: Any = processf
        self.calctype: str = calctype
