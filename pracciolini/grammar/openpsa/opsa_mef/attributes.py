from pracciolini.grammar.openpsa.opsa_mef.base import ValidatableElement
from pracciolini.grammar.openpsa.opsa_mef.identifier import Identifier, NonEmptyString


class Attribute(ValidatableElement):
    def validate(self):
        # Implement validation logic specific to XMLAttribute
        Identifier.validate(self.get("name"))
        NonEmptyString.validate(self.get("value"))
        if self.get("type"):
            NonEmptyString.validate(self.get("type"))


class Attributes(ValidatableElement):
    TAG = "attributes"

    def validate(self):
        # Validate each attribute in the collection
        for attribute in self:
            if not isinstance(attribute, Attribute):
                raise TypeError("All children must be Attribute instances")
            attribute.validate()
