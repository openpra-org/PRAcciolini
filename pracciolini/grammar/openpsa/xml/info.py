from typing import Set


class XMLInfo:
    tag: str = ""
    attrs: Set[str] = set()
    req_attrs: Set[str] = set()
    children: Set[str] = set()
    req_children: Set[str] = set()