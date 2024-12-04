import os
import importlib
from abc import ABC
from typing import Dict, Any

from lxml.etree import Element


def recursive_import(package_dir, package_name):
    for root, dirs, files in os.walk(package_dir):
        # Skip directories that do not contain an __init__.py file (not packages)
        if '__init__.py' not in files and root != package_dir:
            continue

        # Compute relative package name
        rel_path = os.path.relpath(root, package_dir)
        if rel_path == '.':
            rel_package = package_name
        else:
            # Construct package path by replacing path separators with dots
            rel_package = package_name + '.' + '.'.join(rel_path.split(os.sep))

        # Import modules in the current package
        for filename in files:
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]  # Remove .py extension
                full_module_name = rel_package + '.' + module_name
                importlib.import_module(full_module_name)

recursive_import(os.path.dirname(__file__), __name__)

## the registry (OpenPSA Model Exchange)
class OpsaMx(ABC):

    ## A mapping of known class names to classes. Used for tracking all available classes and looking up any given class
    ##
    ## e.g. : BasicEventReference.__name__ -> BasicEventReference.__class__
    class_registry: Dict[str, Any] = {}

    ## map an Element tag to a class. Used to determine the class that should process a given Element. Implies that an
    ## Element with a given tag can only be processed by one class. Note: there is no constraint on what that specific
    ## class might do, or call other classes. Used by `from_xml` and related methods. Quickly check if a tag is parsable
    ##
    ## e.g. : 'define-fault-tree' -> FaultTreeDefinition
    tag_class_registry: Dict[str, Any] = {}

    ## next, what about @meta-tags? for example, we have logical expressions like `or`, `and`, `nor`, etc., where the
    ## underlying logic remains the same, but the tag is different. It is useful to generalize for this case, while still
    ## maintaining flexibility for special-cases, such as `k/n` or `not` (single input) expressions. One way to handle
    ## this is to just declare a MetaClass, say LogicalExpression, and have multiple tags map to it.
    ##
    ## e.g. : 'and' -> LogicalExpression(tag='and').
    ##        'xor' -> LogicalExpression(tag='xor').
    ##
    ## This is fine, but this also means that LogicalExpression will need its own lookup table to handle different cases
    ## One alternative is that we define derived concrete classes, but that can mean a lot of classes, with each class
    ## doing almost nothing interesting (i.e. boilerplate). pain in the ass. this is not Java.
    ##
    ## e.g. : 'and' -> AndExpression extends MultiInputLogicExpression extends LogicExpression
    ##        'not' -> AndExpression extends SingleInputLogicExpression extends LogicExpression



    def __init_subclass__(cls, **kwargs):
        """Automatically called when a subclass is created."""
        super().__init_subclass__(**kwargs)
        # Add the subclass to the registry
        cls.class_registry[cls.__name__] = cls

    @classmethod
    def get_registry(cls):
        """Returns the registry dictionary."""
        return cls.class_registry


    @classmethod
    def from_xml(cls, root: Element):
        # should return an instance of cls
        # return cls() with appropriately instantiated internal attributes
        pass


    def to_xml(self) -> Element:
        ## derived classes build and return an Element, assisted by the base class
        ## final object is returned by the lowest-derived class, where each derived class optionally calls its super's
        ## to_xml() method.
        pass