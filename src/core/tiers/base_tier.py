"""Base class for all tiers"""

from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass
import time


@dataclass
class TierResponse:
    """Response from any tier"""
    tier: int
    text: str
    confidence: float
    generation_time_ms: float
    source: str
    metadata: Dict = None


class BaseTier(ABC):
    """Abstract base class for all tiers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tier_number = None
        self.tier_name = None
    
    @abstractmethod
    def generate(self, query: str, intent: str, context: Dict) -> TierResponse:
        """Generate response for query"""
        pass
    
    def _measure_time(self, func):
        """Measure execution time"""
        start = time.time()
        result = func()
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms


__all__ = ['BaseTier', 'TierResponse']