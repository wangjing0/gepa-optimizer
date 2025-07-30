"""Data models for GEPA optimizer."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CacheEntry:
    """Cache entry for API responses."""
    response: str
    timestamp: float
    usage_count: int = 0

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