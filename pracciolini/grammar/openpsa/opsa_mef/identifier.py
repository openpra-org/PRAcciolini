import re

from pracciolini.grammar.openpsa.opsa_mef.base import ValidatableMixin


class Identifier(ValidatableMixin):
    PATTERN = re.compile(r"[^\-.]+(-[^\-.]+)*")

    @staticmethod
    def validate(value):
        if not Identifier.PATTERN.match(value):
            raise ValueError("Invalid Identifier")


class NonEmptyString(ValidatableMixin):
    @staticmethod
    def validate(value):
        if not value or not value.strip():
            raise ValueError("String cannot be empty or just whitespace")