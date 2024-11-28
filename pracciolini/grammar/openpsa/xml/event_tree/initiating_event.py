from typing import Optional
import lxml
from lxml import etree

from pracciolini.grammar.openpsa.xml.identifier import XMLSerializable


class InitiatingEventDefinition(XMLSerializable):
    """Represents an initiating event definition within an OpenPSA document.

    This class provides an object-oriented representation of an "define-initiating-event" element
    used to define initiating events within OpenPSA documents. It allows for setting the event's name,
    referencing an event tree, and optionally specifying a role, label, and additional attributes.

    Attributes:
        name (str): The unique name for the initiating event.
        event_tree (str): The name of the referenced event tree.
        role (Optional[Role]): An optional Role object associated with the event.
        label (Optional[Label]): An optional Label object associated with the event.
        attributes (Optional[Attributes]): An optional Attributes object containing additional attributes.

    Example:
        ```python
        from pracciolini.grammar.openpsa.xml.identifier import Role, Label

        event_definition = InitiatingEventDefinition(
            name="customer_complaint",
            event_tree="complaint_handling",
            role=Role("customer_service"),
            label=Label("Customer Complaint Received"),
        )
        xml_element = event_definition.to_xml()
        print(etree.tostring(xml_element, pretty_print=True).decode())
        ```

        This example creates an InitiatingEventDefinition object with a name, event tree, role, and label.
        The to_xml() method then converts it into the corresponding XML element structure.
    """

    def __init__(self,
                 name: str,
                 event_tree: Optional[str],
        ) -> None:
        """
        Initializes an InitiatingEventDefinition object.

        Args:
            name (str): The unique name for the initiating event.
            event_tree (Optional[str]): The name of the referenced event tree.
        """

        self.name: str = name
        self.event_tree: Optional[str] = event_tree

    def to_xml(self) -> etree.Element:
        """
        Converts the InitiatingEventDefinition object into a corresponding "define-initiating-event" XML element.

        Returns:
            etree.Element: The constructed XML element representing the initiating event definition.
        """

        element = etree.Element("define-initiating-event")
        element.set("name", self.name)
        if self.event_tree:
            element.set("event-tree", self.event_tree)
        return element

    @classmethod
    def from_xml(cls, root: lxml.etree.Element) -> 'InitiatingEventDefinition':
        """
        Creates an InitiatingEventDefinition object from an XML element.

        Parses the given XML element representing an "define-initiating-event" element and extracts the necessary information
        to create an InitiatingEventDefinition object.

        Args:
            root (lxml.etree.Element): The root XML element to parse.

        Returns:
            InitiatingEventDefinition: The created InitiatingEventDefinition object.

        Raises:
            lxml.etree.ParserError: If the XML element is invalid or missing required attributes.
        """

        if root is None:
            raise lxml.etree.ParserError("Invalid XML element: root cannot be None")

        name = root.get("name")
        if name is None:
            raise lxml.etree.ParserError("Missing required attribute 'name' for initiating-event-definition")

        event_tree = root.get("event-tree")
        return cls(name=name, event_tree=event_tree)

    def __str__(self) -> str:
        """
        Returns a string representation of the InitiatingEventDefinition object.

        The string representation includes the name, event tree, role, label, and attributes of the event definition.
        """

        str_rep = [
            f"define-initiating-event name={self.name}"
        ]
        if self.event_tree:
            str_rep.append(f"event-tree-ref: {self.event_tree}")
        return " ".join(str_rep)