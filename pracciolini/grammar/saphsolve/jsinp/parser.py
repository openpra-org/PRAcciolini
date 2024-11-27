import json
from typing import Any, Dict, List

from pracciolini.grammar.saphsolve.core import Header, TruncParam, EventTree, WorkspacePair, SysGate, FaultTree, Gate
from pracciolini.grammar.saphsolve.core.event import Event
from pracciolini.grammar.saphsolve.core.sequence import Sequence
from pracciolini.grammar.saphsolve.jsinp import JSInp


class JSONParser:
    """
    Parses JSON files into structured objects.

    Attributes:
        file_path (str): Path to the JSON file.
        data (Dict[str, Any]): Parsed JSON data.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initializes a JSONParser instance.

        Args:
            file_path (str): Path to the JSON file.
        """
        self.file_path: str = file_path
        self.data: Dict[str, Any] = self.parse_from_json()

    def parse_from_json(self) -> Dict[str, Any]:
        """
        Parses the JSON file and returns the data.

        Returns:
            Dict[str, Any]: The parsed JSON data.
        """
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def parse_to_object(self) -> JSInp:
        """
        Parses the JSON data into a JSInp object.

        Returns:
            JSInp: The structured JSInp object.
        """
        saphsolve_input = self.data

        # Parse the 'header' section
        header_data = saphsolve_input['saphiresolveinput']['header']
        header = Header(
            projectpath=header_data['projectpath'],
            eventtree=EventTree(
                name=header_data['eventtree']['name'],
                number=header_data['eventtree']['number'],
                initevent=header_data['eventtree']['initevent'],
                seqphase=header_data['eventtree']['seqphase']
            ),
            flagnum=header_data['flagnum'],
            ftcount=header_data['ftcount'],
            fthigh=header_data['fthigh'],
            sqcount=header_data['sqcount'],
            sqhigh=header_data['sqhigh'],
            becount=header_data['becount'],
            behigh=header_data['behigh'],
            mthigh=header_data['mthigh'],
            phhigh=header_data['phhigh'],
            truncparam=TruncParam(
                ettruncopt=header_data['truncparam']['ettruncopt'],
                fttruncopt=header_data['truncparam']['fttruncopt'],
                sizeopt=header_data['truncparam']['sizeopt'],
                ettruncval=header_data['truncparam']['ettruncval'],
                fttruncval=header_data['truncparam']['fttruncval'],
                sizeval=header_data['truncparam']['sizeval'],
                transrepl=header_data['truncparam']['transrepl'],
                transzones=header_data['truncparam']['transzones'],
                translevel=header_data['truncparam']['translevel'],
                usedual=header_data['truncparam']['usedual'],
                dualcutoff=header_data['truncparam']['dualcutoff']
            ),
            workspacepair=WorkspacePair(
                ph=header_data['workspacepair']['ph'],
                mt=header_data['workspacepair']['mt']
            ),
            iworkspacepair=WorkspacePair(
                ph=header_data['iworkspacepair']['ph'],
                mt=header_data['iworkspacepair']['mt']
            )
        )

        # Parse the 'sysgatelist' section
        sysgatelist_data = saphsolve_input['saphiresolveinput']['sysgatelist']
        sysgatelist: List[SysGate] = [
            SysGate(
                name=gate_data['name'],
                id=gate_data['id'],
                gateid=gate_data['gateid'],
                gateorig=gate_data['gateorig'],
                gatepos=gate_data['gatepos'],
                eventid=gate_data['eventid'],
                gatecomp=gate_data['gatecomp'],
                comppos=gate_data['comppos'],
                compflag=gate_data['compflag'],
                gateflag=gate_data['gateflag'],
                gatet=gate_data['gatet'],
                bddsuccess=gate_data['bddsuccess'],
                done=gate_data['done']
            )
            for gate_data in sysgatelist_data
        ]

        # Parse the 'faulttreelist' section
        faulttreelist_data = saphsolve_input['saphiresolveinput']['faulttreelist']
        faulttreelist: List[FaultTree] = []
        for fault in faulttreelist_data:
            if 'gatelist' in fault and fault['gatelist'] is not None:
                gatelist = [
                    Gate(
                        gateid=gate_data['gateid'],
                        gatetype=gate_data['gatetype'],
                        numinputs=gate_data['numinputs'],
                        gateinput=gate_data.get('gateinput'),
                        eventinput=gate_data.get('eventinput'),
                        compeventinput=gate_data.get('compeventinput')
                    )
                    for gate_data in fault['gatelist']
                ]
            else:
                gatelist = None
            fault_tree = FaultTree(
                ftheader=fault['ftheader'],
                gatelist=gatelist
            )
            faulttreelist.append(fault_tree)

        # Parse the 'sequencelist' section if it exists
        if 'sequencelist' in saphsolve_input['saphiresolveinput']:
            sequencelist_data = saphsolve_input['saphiresolveinput']['sequencelist']
            sequencelist: List[Sequence] = [
                Sequence(
                    seqid=sequence_data['seqid'],
                    etid=sequence_data['etid'],
                    initid=sequence_data['initid'],
                    qmethod=sequence_data['qmethod'],
                    qpasses=sequence_data['qpasses'],
                    numlogic=sequence_data['numlogic'],
                    blocksize=sequence_data['blocksize'],
                    logiclist=sequence_data['logiclist']
                )
                for sequence_data in sequencelist_data
            ]
        else:
            sequencelist = []

        # Parse the 'eventlist' section
        eventlist_data = saphsolve_input['saphiresolveinput']['eventlist']
        eventlist: List[Event] = [
            Event(
                id=event_data['id'],
                corrgate=event_data['corrgate'],
                name=event_data['name'],
                evworkspacepair=event_data['evworkspacepair'],
                value=event_data['value'],
                initf=event_data['initf'],
                processf=event_data['processf'],
                calctype=event_data['calctype']
            )
            for event_data in eventlist_data
        ]

        # Create the JSInp object
        saphsolve_input_object = JSInp(
            version=saphsolve_input['version'],
            saphiresolveinput={
                'header': header,
                'sysgatelist': sysgatelist,
                'faulttreelist': faulttreelist,
                'sequencelist': sequencelist,
                'eventlist': eventlist
            }
        )

        return saphsolve_input_object
