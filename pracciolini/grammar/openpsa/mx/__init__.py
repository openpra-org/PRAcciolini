import os
import importlib


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