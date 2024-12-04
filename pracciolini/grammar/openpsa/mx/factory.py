from pracciolini.grammar.openpsa.mx.registry import XMLRegistry
from lxml.etree import Element

def build_from_xml(element: Element):
    tag = element.tag
    cls = XMLRegistry.get_class_by_tag(tag)
    if cls is None:
        raise ValueError(f"No class registered for tag '{tag}'")
    return cls.from_xml(element)