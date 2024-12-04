from pracciolini.grammar.openpsa.mx.registry import XMLRegistry

def xml_register(*tags):
    def decorator(class_obj):
        XMLRegistry.register(class_obj, tags)
        return class_obj
    return decorator