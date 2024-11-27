from typing import Any


class WorkspacePair:
    """
    Represents a pair of workspaces.

    Attributes:
        ph (Any): PH workspace.
        mt (Any): MT workspace.
    """

    def __init__(self, ph: Any, mt: Any) -> None:
        """
        Initializes a WorkspacePair instance.

        Args:
            ph (Any): PH workspace.
            mt (Any): MT workspace.
        """
        self.ph: Any = ph
        self.mt: Any = mt
