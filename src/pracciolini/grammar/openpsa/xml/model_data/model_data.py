from pracciolini.grammar.openpsa.xml.expression.meta import ModelDataDefinitionsMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable


class ModelData(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="model-data",
                                 class_type=self,
                                 children=ModelDataDefinitionsMeta.permitted_tags)
        super().__init__(*args, **kwargs)
