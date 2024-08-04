import networkx as nx
from typing import Any, Optional, Dict

from pracciolini.core.grammar import Grammar
from pracciolini.core.translation import Translation


class GrammarRegistry:
    """
    A registry for managing grammars and translations between them using a directed graph.

    This class uses a directed multigraph from the NetworkX library to store grammars as nodes
    and translations as edges between these nodes.

    Attributes:
        graph (nx.MultiDiGraph): The directed multigraph that stores grammars and translations.

    Methods:
        add_grammar: Adds a grammar to the graph as a node.
        add_translation: Adds a translation between two grammars in the graph as an edge.
        get_translations: Retrieves translation data between two specified grammars.
        translate: Translates data from one grammar to another using a specified translation.
    """

    graph = nx.MultiDiGraph()

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

        # Ensure grammars are added to the registry. If already present, replaces existing grammars.
        GrammarRegistry.graph.add_node(from_grammar)
        GrammarRegistry.graph.add_node(to_grammar)
        GrammarRegistry.graph.add_edge(from_grammar, to_grammar, key=translation.key, translation=translation)

    @staticmethod
    def get_translations(from_grammar: Grammar, to_grammar: Grammar) -> Optional[Dict[str, Any]]:
        """
        Retrieves all translation data between two specified grammars.

        Args:
            from_grammar (Grammar): The source grammar.
            to_grammar (Grammar): The target grammar.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing translation data if available, otherwise None.
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
