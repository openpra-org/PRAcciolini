from typing import Dict, Any


class JSInp:
    """
    Represents the JSON input structure.

    Attributes:
        version (str): Version of the input.
        saphiresolveinput (Dict[str, Any]): Detailed saphiresolve input data.
    """

    def __init__(self, version: str, saphiresolveinput: Dict[str, Any]) -> None:
        """
        Initializes a JSInp instance.

        Args:
            version (str): Version of the input.
            saphiresolveinput (Dict[str, Any]): Detailed saphiresolve input data.
        """
        self.version: str = version
        self.saphiresolveinput: Dict[str, Any] = saphiresolveinput
