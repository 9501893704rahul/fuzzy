"""
Fuzzy Rule-Based Classification System
A research project implementing Fuzzy Rule-Based Classification with automatic 
rule generation and genetic algorithm optimization for interpretable medical diagnosis.
"""

from .membership_functions import MembershipFunctionManager
from .rule_generation import RuleGenerator
from .genetic_optimizer import GeneticOptimizer
from .fuzzy_classifier import FuzzyRuleClassifier

__all__ = [
    'MembershipFunctionManager',
    'RuleGenerator', 
    'GeneticOptimizer',
    'FuzzyRuleClassifier'
]

__version__ = '1.0.0'
