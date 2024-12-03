from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class InitialStateDefinition(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="initial-state",
                                 class_type=self,
                                 children={"fork", "sequence", "rule", "collect-expression", "branch"})
        super().__init__(*args, **kwargs)
