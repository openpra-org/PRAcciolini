from typing import Set


class Meta:
    pass


class ModelDataDefinitionsMeta(Meta):
    permitted_tags: Set[str] = {
        "define-basic-event",
        "define-house-event",
        "define-parameter",
        "define-CCF-group",
        "define-component"
    }

class EventDefinitionsMeta(Meta):
    permitted_tags: Set[str] = {
        "define-gate",
        "define-extern-function",
        "define-extern-library",
    }.union(ModelDataDefinitionsMeta.permitted_tags)


class DistributionMeta(Meta):
    permitted_tags: Set[str] = {
        "Weibull",
        "uniform-deviate",
        "lognormal-deviate",
        "exponential",
        "GLM",
        "normal-deviate",
        "gamma-deviate",
        "beta-deviate",
        "histogram",
        "periodic-test"
    }


class ConstantsMeta(Meta):
    permitted_tags: Set[str] = {
        "float",
        "int",
        "bool",
        "constant",
        "pi"
    }


class ArithmeticMeta(Meta):
    permitted_tags: Set[str] = {
        "add",
        "sub",
        "mul",
        "div",
        "abs",
        "sin",
        "cos",
        "sinh",
        "cosh",
        "neg",
    }


class LogicalUnaryMeta(Meta):
    permitted_tags: Set[str] = {
        "not",
        "iff",
    }


class LogicalMultiplicativeMeta(Meta):
    permitted_tags: Set[str] = {
        "and",
        "or",
        "xor",
        "nor",
        "xnor",
        "nand",
        "imply",
        "atleast",
        "cardinality",
    }


class LogicalMeta(Meta):
    permitted_tags: Set[str] = (set()
                                .union(LogicalUnaryMeta.permitted_tags)
                                .union(LogicalMultiplicativeMeta.permitted_tags))


class ReferenceMeta(Meta):
    permitted_tags: Set[str] = {
        "parameter",
        "system-mission-time",
        "extern-function",
        "basic-event",
        "house-event",
        "event",
        "gate",
    }

class ExpressionMeta(Meta):
    permitted_tags: Set[str] = (set()
                                .union(ConstantsMeta.permitted_tags)
                                .union(DistributionMeta.permitted_tags)
                                .union(ArithmeticMeta.permitted_tags)
                                .union(LogicalMeta.permitted_tags)
                                .union(ReferenceMeta.permitted_tags))


