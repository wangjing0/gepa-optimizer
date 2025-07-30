"""Data models for GEPA optimization."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Task:
    """Training task with input and expected criteria."""
    input: str
    output: Optional[List[str]]


@dataclass
class Candidate:
    """Prompt candidate with evaluation results."""
    id: int
    prompt: str
    parent_id: Optional[int]
    scores: List[float]
    avg_score: float