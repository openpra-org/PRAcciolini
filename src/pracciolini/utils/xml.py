from lxml.etree import Element


def deep_compare_xml(xml_a: Element, xml_b: Element) -> bool:

    if xml_a is None and xml_b is None:
        return True

    if xml_a is None or xml_b is None:
        raise ValueError("cannot compare with NoneType")

    if not hasattr(xml_a, "tag") or not hasattr(xml_b, "tag"):
        raise ValueError("cannot compare when elements don't have tags")

    #print(f"comparing tags {xml_a.tag} and {xml_b.tag}")

    if getattr(xml_a, "tag") != getattr(xml_b, "tag"):
        raise ValueError(f"re-serialized xml tags do not match: {xml_a.tag} != {xml_b.tag}")

    if hasattr(xml_a, "attrib") ^ hasattr(xml_b, "attrib"):
        raise ValueError("cannot compare when one element does not have attrib")

    if xml_a.attrib != xml_b.attrib:
        raise ValueError(f"attributes [{xml_a.attrib}] do not match [{xml_b.attrib}]")

    #print(f"comparing [{xml_a.tag}:{xml_a.attrib}] and [{xml_b.tag}:{xml_b.attrib}]")

    child_tags_a = sorted(element.tag for element in xml_a)
    child_tags_b = sorted(element.tag for element in xml_b)
    if child_tags_a != child_tags_b:
        raise ValueError(f"child tags <{xml_a.tag}/>:{child_tags_a} do not match <{xml_b.tag}>:{child_tags_b}, extra: [{set(child_tags_a) - set(child_tags_b)}]")

    children_a = (element for element in xml_a)
    children_b = (element for element in xml_b)

    all_match_so_far = True
    for child_a, child_b in zip(children_a, children_b):
        this_one_matches = deep_compare_xml(child_a, child_b)
        all_match_so_far = all_match_so_far and this_one_matches

    return all_match_so_far
