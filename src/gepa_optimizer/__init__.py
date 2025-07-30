"""GEPA: Genetic-Evolutionary Prompt Architecture

Automated prompt optimization using LLM feedback loops with Claude API integration.
"""

from .core import GEPAOptimizer, run_gepa_optimization, run_gepa_optimization_async
from .models import Task, Candidate, CacheEntry

__version__ = "0.1.0"
__author__ = "Jing Wang"

__all__ = [
    "GEPAOptimizer",
    "run_gepa_optimization",
    "run_gepa_optimization_async",
    "Task",
    "Candidate",
]