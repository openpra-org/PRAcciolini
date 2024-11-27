from typing import Any, Optional, List

from pracciolini.grammar.saphsolve.core import Gate


class FaultTree:
    """
    Represents a fault tree.

    Attributes:
        ftheader (Any): Header information for the fault tree.
        gatelist (Optional[List[Gate]]): List of gates in the fault tree.
    """

    def __init__(self, ftheader: Any, gatelist: Optional[List[Gate]]) -> None:
        """
        Initializes a FaultTree instance.

        Args:
            ftheader (Any): Header information for the fault tree.
            gatelist (Optional[List[Gate]]): List of gates in the fault tree.
        """
        self.ftheader: Any = ftheader
        self.gatelist: Optional[List[Gate]] = gatelist
