from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class ModelData(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="model-data",
                                 class_type=self,
                                 children={"define-basic-event", "define-house-event", "define-parameter"})
        super().__init__(*args, **kwargs)
