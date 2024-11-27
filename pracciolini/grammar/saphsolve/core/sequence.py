from typing import Any, List


class Sequence:
    """
    Represents a sequence in the system.

    Attributes:
        seqid (int): Sequence identifier.
        etid (int): ET identifier.
        initid (int): Initial identifier.
        qmethod (str): Quantification method.
        qpasses (int): Number of quantification passes.
        numlogic (int): Number of logical operations.
        blocksize (int): Size of the block.
        logiclist (List[Any]): List of logical operations.
    """

    def __init__(
        self,
        seqid: int,
        etid: int,
        initid: int,
        qmethod: str,
        qpasses: int,
        numlogic: int,
        blocksize: int,
        logiclist: List[Any]
    ) -> None:
        """
        Initializes a Sequence instance.

        Args:
            seqid (int): Sequence identifier.
            etid (int): ET identifier.
            initid (int): Initial identifier.
            qmethod (str): Quantification method.
            qpasses (int): Number of quantification passes.
            numlogic (int): Number of logical operations.
            blocksize (int): Size of the block.
            logiclist (List[Any]): List of logical operations.
        """
        self.seqid: int = seqid
        self.etid: int = etid
        self.initid: int = initid
        self.qmethod: str = qmethod
        self.qpasses: int = qpasses
        self.numlogic: int = numlogic
        self.blocksize: int = blocksize
        self.logiclist: List[Any] = logiclist
