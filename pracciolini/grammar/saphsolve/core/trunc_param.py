from typing import Any


class TruncParam:
    """
    Represents truncation parameters.

    Attributes:
        ettruncopt (Any): Truncation option for ET.
        fttruncopt (Any): Truncation option for FT.
        sizeopt (Any): Size option.
        ettruncval (Any): Truncation value for ET.
        fttruncval (Any): Truncation value for FT.
        sizeval (Any): Size value.
        transrepl (Any): Transaction replace parameter.
        transzones (Any): Transaction zones parameter.
        translevel (Any): Transaction level parameter.
        usedual (bool): Flag indicating whether dual usage is enabled.
        dualcutoff (Any): Dual cutoff value.
    """

    def __init__(
        self,
        ettruncopt: Any,
        fttruncopt: Any,
        sizeopt: Any,
        ettruncval: Any,
        fttruncval: Any,
        sizeval: Any,
        transrepl: Any,
        transzones: Any,
        translevel: Any,
        usedual: bool,
        dualcutoff: Any
    ) -> None:
        """
        Initializes a TruncParam instance.

        Args:
            ettruncopt (Any): Truncation option for ET.
            fttruncopt (Any): Truncation option for FT.
            sizeopt (Any): Size option.
            ettruncval (Any): Truncation value for ET.
            fttruncval (Any): Truncation value for FT.
            sizeval (Any): Size value.
            transrepl (Any): Transaction replace parameter.
            transzones (Any): Transaction zones parameter.
            translevel (Any): Transaction level parameter.
            usedual (bool): Flag indicating whether dual usage is enabled.
            dualcutoff (Any): Dual cutoff value.
        """
        self.ettruncopt: Any = ettruncopt
        self.fttruncopt: Any = fttruncopt
        self.sizeopt: Any = sizeopt
        self.ettruncval: Any = ettruncval
        self.fttruncval: Any = fttruncval
        self.sizeval: Any = sizeval
        self.transrepl: Any = transrepl
        self.transzones: Any = transzones
        self.translevel: Any = translevel
        self.usedual: bool = usedual
        self.dualcutoff: Any = dualcutoff
