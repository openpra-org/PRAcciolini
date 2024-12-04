from lxml import etree

from pracciolini.grammar.openpsa.mx.define import EventDefinition
from pracciolini.grammar.openpsa.mx.factory import build_from_xml

if __name__ == "__main__":

    event = EventDefinition(name='My Event')
    xml_element = event.to_xml()
    xml_string = etree.tostring(xml_element, pretty_print=True)
    print(xml_string.decode())


    xml_string = '<and name="My Event"></and>'
    xml_element = etree.fromstring(xml_string)
    event = build_from_xml(xml_element)
    print(event)