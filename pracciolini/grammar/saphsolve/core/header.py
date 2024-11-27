import json
from typing import Any, Dict

from pracciolini.grammar.saphsolve.core.event_tree import EventTree
from pracciolini.grammar.saphsolve.core.trunc_param import TruncParam
from pracciolini.grammar.saphsolve.core.workspace_pair import WorkspacePair

class Header:
    """
    Represents the header section of the input data.

    Attributes:
        projectpath (str): The project path.
        eventtree (EventTree): The associated event tree.
        flagnum (int): Number of flags.
        ftcount (int): FT count.
        fthigh (Any): FT high value.
        sqcount (int): SQ count.
        sqhigh (Any): SQ high value.
        becount (int): BE count.
        behigh (Any): BE high value.
        mthigh (Any): MT high value.
        phhigh (Any): PH high value.
        truncparam (TruncParam): Truncation parameters.
        workspacepair (WorkspacePair): Workspace pair.
        iworkspacepair (WorkspacePair): Inverse workspace pair.
    """

    def __init__(
        self,
        projectpath: str,
        eventtree: EventTree,
        flagnum: int,
        ftcount: int,
        fthigh: Any,
        sqcount: int,
        sqhigh: Any,
        becount: int,
        behigh: Any,
        mthigh: Any,
        phhigh: Any,
        truncparam: TruncParam,
        workspacepair: WorkspacePair,
        iworkspacepair: WorkspacePair
    ) -> None:
        """
        Initializes a Header instance.

        Args:
            projectpath (str): The project path.
            eventtree (EventTree): The associated event tree.
            flagnum (int): Number of flags.
            ftcount (int): FT count.
            fthigh (Any): FT high value.
            sqcount (int): SQ count.
            sqhigh (Any): SQ high value.
            becount (int): BE count.
            behigh (Any): BE high value.
            mthigh (Any): MT high value.
            phhigh (Any): PH high value.
            truncparam (TruncParam): Truncation parameters.
            workspacepair (WorkspacePair): Workspace pair.
            iworkspacepair (WorkspacePair): Inverse workspace pair.
        """
        self.projectpath: str = projectpath
        self.eventtree: EventTree = eventtree
        self.flagnum: int = flagnum
        self.ftcount: int = ftcount
        self.fthigh: Any = fthigh
        self.sqcount: int = sqcount
        self.sqhigh: Any = sqhigh
        self.becount: int = becount
        self.behigh: Any = behigh
        self.mthigh: Any = mthigh
        self.phhigh: Any = phhigh
        self.truncparam: TruncParam = truncparam
        self.workspacepair: WorkspacePair = workspacepair
        self.iworkspacepair: WorkspacePair = iworkspacepair

    def to_json(self) -> json | Dict[str, Any]:
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
            'projectpath': self.projectpath,
            'eventtree': self.eventtree.__dict__,
            'flagnum': self.flagnum,
            'ftcount': self.ftcount,
            'fthigh': self.fthigh,
            'sqcount': self.sqcount,
            'sqhigh': self.sqhigh,
            'becount': self.becount,
            'behigh': self.behigh,
            'mthigh': self.mthigh,
            'phhigh': self.phhigh,
            'truncparam': self.truncparam.__dict__,
            'workspacepair': self.workspacepair.__dict__,
            'iworkspacepair': self.iworkspacepair.__dict__
        }
        return {key: val for key, val in header_dict.items() if val is not None}
