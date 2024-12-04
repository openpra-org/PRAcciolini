

class OpsaMef:
    registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically called when a subclass is created."""
        super().__init_subclass__(**kwargs)
        # Add the subclass to the registry
        cls.registry[cls.__name__] = cls

    @classmethod
    def get_registry(cls):
        """Returns the registry dictionary."""
        return cls.registry