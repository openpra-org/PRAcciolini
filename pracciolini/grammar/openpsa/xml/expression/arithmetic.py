from typing import Set, Optional, Dict, Union

from lxml import etree
from lxml.etree import Element

from pracciolini.grammar.openpsa.xml.expression.meta import ExpressionMeta, ArithmeticMeta
from pracciolini.grammar.openpsa.xml.serializable import XMLSerializable, XMLInfo


class ConstructedArithmeticExp:
    pass


class ArithmeticExpression(XMLSerializable):

    registered: bool = False
    def __init__(self, *args, **kwargs) -> None:
        #print("in constructor ArithmeticExpression", kwargs)
        # if "tag" in kwargs:
        #     kwargs["info"] = XMLInfo(tag=kwargs["tag"], class_type=self, children=ExpressionMeta.permitted_tags)
        #     super().__init__(*args, **kwargs)
        #     return
        all_tags: Set[str] = ArithmeticMeta.permitted_tags
        if not ArithmeticExpression.registered:
            ArithmeticExpression.registered = True
            for tag in all_tags:
                kwargs["info"] = XMLInfo(tag=tag, class_type=self, children=ExpressionMeta.permitted_tags)
                super().__init__(*args, **kwargs)

        kwargs["info"] = XMLInfo(tag="meta-arithmetic", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)

    # def to_xml(self, tag: Optional[Union[XMLInfo, str]] = None) -> etree.Element:
    #     return super().to_xml(tag)

    #
    # def to_xml(self) -> etree.Element:
    #     return super().to_xml()

    # @staticmethod
    # def static_to_xml(instance: 'ArithmeticExpression', info: XMLInfo) -> etree.Element:
    #     #return super().to_xml()

    # @classmethod
    # def from_xml(cls, root: Element):
    #     info: XMLInfo = XMLInfo.get(root.tag)
    #     if hasattr(cls, "info"):
    #         print(f"cls.info.tag:{cls.info.tag}, root.tag:{root.tag}, registry.info.tag:{info.tag}")
    #     else:
    #         print(f"cls.info.tag:None, root.tag: {root.tag}, registry.info.tag: {info.tag}")
    #     cls_instance = super().from_xml(root)
    #     if hasattr(cls_instance, "info"):
    #         print(f"cls.info.tag:{cls_instance.info.tag}, root.tag:{root.tag}, registry.info.tag:{info.tag}")
    #     else:
    #         print(f"cls.info.tag:None, root.tag: {root.tag}, registry.info.tag: {info.tag}")
    #     return cls_instance

    # @classmethod
    # def validate(cls, instance: 'XMLSerializable'):
    #     return super().validate(instance)
    #
    # @staticmethod
    # def build(root: Element):
    #     return super().build(root)


class ArithmeticAddExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="add", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArithmeticAbs(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="abs", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArithmeticSin(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="sin", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArithmeticNegativeExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="neg", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArthmeticSubtractExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="sub", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArthmeticMultiplyExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="mul", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)


class ArthmeticDivideExpression(XMLSerializable):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["info"] = XMLInfo(tag="div", class_type=self, children=ExpressionMeta.permitted_tags)
        super().__init__(*args, **kwargs)
