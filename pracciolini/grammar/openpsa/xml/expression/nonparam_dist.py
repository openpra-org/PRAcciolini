from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class HistogramBin(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="bin", class_type=self, children={"float"}, req_children={"float"})
        super().__init__(*args, **kwargs)


class Histogram(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="histogram", class_type=self, children={"float", "bin"})
        super().__init__(*args, **kwargs)

