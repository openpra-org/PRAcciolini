from lxml.etree import Element

from pracciolini.grammar.openpsa.xml.attributes import Attributes, Attribute
from pracciolini.grammar.openpsa.xml.event_tree.branch import BranchDefinition, BranchReference
from pracciolini.grammar.openpsa.xml.event_tree.event_tree import EventTreeReference, EventTreeDefinition
from pracciolini.grammar.openpsa.xml.event_tree.fork import ForkDefinition, PathDefinition
from pracciolini.grammar.openpsa.xml.event_tree.functional_event import FunctionalEventDefinition
from pracciolini.grammar.openpsa.xml.event_tree.initial_state import InitialStateDefinition
from pracciolini.grammar.openpsa.xml.event_tree.initiating_event import InitiatingEventDefinition
from pracciolini.grammar.openpsa.xml.event_tree.sequence import SequenceDefinition, SequenceReference
from pracciolini.grammar.openpsa.xml.expression.collect import CollectExpression, CollectFormula
from pracciolini.grammar.openpsa.xml.expression.constants import ConstantExpression, ConstantPi,  FloatExpression, IntegerExpression, BoolExpression
from pracciolini.grammar.openpsa.xml.define_event import BasicEventDefinition, HouseEventDefinition, \
    ParameterDefinition
from pracciolini.grammar.openpsa.xml.expression.arithmetic import ArithmeticAddExpression, ArithmeticNegativeExpression, ArthmeticSubtractExpression, ArthmeticMultiplyExpression, \
    ArthmeticDivideExpression
from pracciolini.grammar.openpsa.xml.expression.logical import RuleDefinition, RuleReference, CardinalityExpression, \
    AtleastExpression, NotExpression
from pracciolini.grammar.openpsa.xml.expression.nonparam_dist import HistogramBin, Histogram
from pracciolini.grammar.openpsa.xml.expression.param_dist import WeibullExpression, LognormalDeviateExpression, \
    UniformDeviateExpression, ExponentialExpression, GLMExpression, NormalDeviateExpression, \
    GammaDeviateExpression, BetaDeviateExpression, PeriodicTestExpression
from pracciolini.grammar.openpsa.xml.fault_tree.event_reference import BasicEventReference, GateReference
from pracciolini.grammar.openpsa.xml.identifier import Label
from pracciolini.grammar.openpsa.xml.model_data.model_data import ModelData
from pracciolini.grammar.openpsa.xml.openpsa_mef import OpsaMef
from pracciolini.grammar.openpsa.xml.reference import SystemMissionTimeParameterReference, ExternalFunctionReference, ParameterReference
from pracciolini.grammar.openpsa.xml.serializable import XMLInfo, XMLSerializable
from pracciolini.grammar.openpsa.xml.singleton import Singleton


@Singleton
class OpsaMefXmlRegistry:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ## root
        XMLInfo.register(OpsaMef)

        ## event tree
        XMLInfo.register(EventTreeDefinition)
        XMLInfo.register(EventTreeReference)
        XMLInfo.register(InitiatingEventDefinition)
        XMLInfo.register(FunctionalEventDefinition)
        XMLInfo.register(SequenceDefinition)
        XMLInfo.register(SequenceReference)
        XMLInfo.register(CollectExpression)
        XMLInfo.register(InitialStateDefinition)
        XMLInfo.register(ForkDefinition)
        XMLInfo.register(PathDefinition)
        XMLInfo.register(CollectFormula)
        XMLInfo.register(BranchDefinition)
        XMLInfo.register(BranchReference)


        ## higher-level
        XMLInfo.register(ModelData)

        ## definitions
        XMLInfo.register(BasicEventDefinition)
        XMLInfo.register(HouseEventDefinition)
        XMLInfo.register(ParameterDefinition)

        ## identifiers
        XMLInfo.register(Attributes)
        XMLInfo.register(Attribute)
        XMLInfo.register(Label)

        ## references
        XMLInfo.register(SystemMissionTimeParameterReference)
        XMLInfo.register(ExternalFunctionReference)
        XMLInfo.register(ParameterReference)
        XMLInfo.register(GateReference)
        XMLInfo.register(BasicEventReference)

        ## constants
        XMLInfo.register(ConstantExpression)
        XMLInfo.register(FloatExpression)
        XMLInfo.register(IntegerExpression)
        XMLInfo.register(BoolExpression)
        XMLInfo.register(ConstantPi)

        ## distributions
        XMLInfo.register(WeibullExpression)
        XMLInfo.register(LognormalDeviateExpression)
        XMLInfo.register(NormalDeviateExpression)
        XMLInfo.register(BetaDeviateExpression)
        XMLInfo.register(GammaDeviateExpression)
        XMLInfo.register(UniformDeviateExpression)
        XMLInfo.register(ExponentialExpression)
        XMLInfo.register(GLMExpression)

        ## expressions
        XMLInfo.register(ArithmeticAddExpression)
        XMLInfo.register(ArithmeticNegativeExpression)
        XMLInfo.register(ArthmeticSubtractExpression)
        XMLInfo.register(ArthmeticMultiplyExpression)
        XMLInfo.register(ArthmeticDivideExpression)

        ## logical
        XMLInfo.register(RuleDefinition)
        XMLInfo.register(RuleReference)
        XMLInfo.register(CardinalityExpression)
        XMLInfo.register(AtleastExpression)
        XMLInfo.register(NotExpression)

        ## exotic
        XMLInfo.register(PeriodicTestExpression)
        XMLInfo.register(HistogramBin)
        XMLInfo.register(Histogram)

    @staticmethod
    def build(root: Element):
        return XMLSerializable.build(root)


