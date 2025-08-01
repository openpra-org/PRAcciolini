import networkx as nx
from typing import Any, Dict, List, Mapping

from pracciolini.core.grammar import Grammar
from pracciolini.core.translation import Translation
from pracciolini.core.validation import Validation
from pracciolini.core.loading import Loader
from pracciolini.core.saving import Saver


class GrammarRegistry:
    """
    A registry for managing grammars, translations between them, validations, loaders, and savers.

    This class uses a directed multigraph from the NetworkX library to store grammars as nodes
    and translations as edges between these nodes. It also maintains separate mappings for
    validations, loaders, and savers associated with each grammar.

    Attributes:
        graph (nx.MultiDiGraph): The directed multigraph that stores grammars and translations.
        validations (Dict[Grammar, List[Validation]]): A mapping of grammars to their validations.
        loaders (Dict[Grammar, Dict[str, List[Loader]]]): A mapping of grammars to a dictionary of extensions and their loaders.
        savers (Dict[Grammar, Dict[str, List[Saver]]]): A mapping of grammars to a dictionary of extensions and their savers.

    Methods:
        add_grammar: Adds a grammar to the graph as a node.
        add_translation: Adds a translation between two grammars in the graph as an edge.
        get_translations: Retrieves translation data between two specified grammars.
        translate: Translates data from one grammar to another using a specified translation.
        add_validation: Adds a validation to a grammar.
        get_validations: Retrieves validations associated with a grammar.
        add_loader: Adds a loader to a grammar for a specific file extension.
        get_loaders: Retrieves loaders associated with a grammar and file extension.
        add_saver: Adds a saver to a grammar for a specific file extension.
        get_savers: Retrieves savers associated with a grammar and file extension.
    """

    graph = nx.MultiDiGraph()
    validations: Dict[Grammar, List[Validation]] = {}
    loaders: Dict[Grammar, Dict[str, List[Loader]]] = {}
    savers: Dict[Grammar, Dict[str, List[Saver]]] = {}

    @staticmethod
    def add_grammar(grammar: Grammar) -> None:
        """
        Adds a grammar to the registry graph as a node.

        Args:
            grammar (Grammar): The grammar to be added to the graph.
        """
        GrammarRegistry.graph.add_node(grammar)

    @staticmethod
    def add_translation(from_grammar: Grammar, to_grammar: Grammar, translation: Translation) -> None:
        """
        Adds a translation between two grammars to the registry graph as an edge.

        Args:
            from_grammar (Grammar): The source grammar from which the translation starts.
            to_grammar (Grammar): The target grammar to which the translation goes.
            translation (Translation): The translation object containing the translation details.
        """

        # Ensure grammars are added to the registry.
        GrammarRegistry.add_grammar(from_grammar)
        GrammarRegistry.add_grammar(to_grammar)
        GrammarRegistry.graph.add_edge(
            from_grammar, to_grammar, key=translation.key, translation=translation
        )

    @staticmethod
    def get_translations(from_grammar: Grammar, to_grammar: Grammar) -> Mapping[str, Any]:
        """
        Retrieves all translation data between two specified grammars.

        Args:
            from_grammar (Grammar): The source grammar.
            to_grammar (Grammar): The target grammar.

        Returns:
            Mapping[str, Any]: A dictionary containing translation data if available, otherwise None.
        """
        return GrammarRegistry.graph.get_edge_data(from_grammar, to_grammar)

    @staticmethod
    def translate(from_grammar: Grammar, to_grammar: Grammar, translation_key: str, data: Any) -> Any:
        """
        Translates data from one grammar to another using a specified translation.

        Args:
            from_grammar (Grammar): The grammar from which the data originates.
            to_grammar (Grammar): The grammar to which the data should be translated.
            translation_key (str): The key identifying the specific translation to use.
            data (Any): The data to be translated.

        Returns:
            Any: The translated data.

        Raises:
            ValueError: If the translation is not found or the translation key is invalid.
        """
        edge_data = GrammarRegistry.graph.get_edge_data(from_grammar, to_grammar)
        if edge_data:
            for key, value in edge_data.items():
                if key == translation_key:
                    return value['translation'].translate_func(data)
        raise ValueError("Translation not found or invalid translation key")

    @staticmethod
    def add_validation(grammar: Grammar, validation: Validation) -> None:
        """
        Adds a validation function to a grammar in the registry.

        Args:
            grammar (Grammar): The grammar to which the validation is associated.
            validation (Validation): The validation object containing the validation details.
        """
        # Ensure the grammar node exists
        GrammarRegistry.add_grammar(grammar)

        # Add the validation to the grammar's list of validations
        if grammar not in GrammarRegistry.validations:
            GrammarRegistry.validations[grammar] = []
        GrammarRegistry.validations[grammar].append(validation)

    @staticmethod
    def get_validations(grammar: Grammar) -> List[Validation]:
        """
        Retrieves all validations associated with a specified grammar.

        Args:
            grammar (Grammar): The grammar whose validations are to be retrieved.

        Returns:
            List[Validation]: A list of validation objects associated with the grammar.
        """
        return GrammarRegistry.validations.get(grammar, [])

    @staticmethod
    def add_loader(grammar: Grammar, extension: str, loader: Loader) -> None:
        """
        Adds a loader to a grammar for a specific file extension.

        Args:
            grammar (Grammar): The grammar for which the loader is defined.
            extension (str): The file extension that the loader can handle.
            loader (Loader): The loader object containing the loader details.
        """
        # Ensure the grammar node exists
        GrammarRegistry.add_grammar(grammar)

        # Initialize the loaders dictionary for the grammar if not present
        if grammar not in GrammarRegistry.loaders:
            GrammarRegistry.loaders[grammar] = {}

        # Initialize the list of loaders for the extension if not present
        if extension not in GrammarRegistry.loaders[grammar]:
            GrammarRegistry.loaders[grammar][extension] = []

        # Add the loader to the list
        GrammarRegistry.loaders[grammar][extension].append(loader)

    @staticmethod
    def get_loaders(grammar: Grammar, extension: str) -> List[Loader]:
        """
        Retrieves loaders associated with a grammar and file extension.

        Args:
            grammar (Grammar): The grammar whose loaders are to be retrieved.
            extension (str): The file extension to filter loaders.

        Returns:
            List[Loader]: A list of loader objects associated with the grammar and extension.
        """
        return GrammarRegistry.loaders.get(grammar, {}).get(extension, [])

    @staticmethod
    def add_saver(grammar: Grammar, extension: str, saver: Saver) -> None:
        """
        Adds a saver to a grammar for a specific file extension.

        Args:
            grammar (Grammar): The grammar for which the saver is defined.
            extension (str): The file extension that the saver can handle.
            saver (Saver): The saver object containing the saver details.
        """
        # Ensure the grammar node exists
        GrammarRegistry.add_grammar(grammar)

        # Initialize the savers dictionary for the grammar if not present
        if grammar not in GrammarRegistry.savers:
            GrammarRegistry.savers[grammar] = {}

        # Initialize the list of savers for the extension if not present
        if extension not in GrammarRegistry.savers[grammar]:
            GrammarRegistry.savers[grammar][extension] = []

        # Add the saver to the list
        GrammarRegistry.savers[grammar][extension].append(saver)

    @staticmethod
    def get_savers(grammar: Grammar, extension: str) -> List[Saver]:
        """
        Retrieves savers associated with a grammar and file extension.

        Args:
            grammar (Grammar): The grammar whose savers are to be retrieved.
            extension (str): The file extension to filter savers.

        Returns:
            List[Saver]: A list of saver objects associated with the grammar and extension.
        """
        return GrammarRegistry.savers.get(grammar, {}).get(extension, [])
