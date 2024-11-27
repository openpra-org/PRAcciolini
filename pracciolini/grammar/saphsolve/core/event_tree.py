from typing import Any


class EventTree:
    """
    Represents an event tree structure.

    Attributes:
        name (str): The name of the event tree.
        number (int): The identifier number of the event tree.
        initevent (Any): The initial event associated with the event tree.
        seqphase (Any): The sequence phase of the event tree.
    """

    def __init__(self, name: str, number: int, initevent: Any, seqphase: Any) -> None:
        """
        Initializes an EventTree instance.

        Args:
            name (str): The name of the event tree.
            number (int): The identifier number of the event tree.
            initevent (Any): The initial event associated with the event tree.
            seqphase (Any): The sequence phase of the event tree.
        """
        self.name: str = name
        self.number: int = number
        self.initevent: Any = initevent
        self.seqphase: Any = seqphase
