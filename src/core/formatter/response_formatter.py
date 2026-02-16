"""Response Formatter using Pydantic."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime


class FormattedResponse(BaseModel):
    """Structured response with validation"""

    response_id: str = Field(..., description="Unique response ID")
    query: str = Field(..., min_length=1, max_length=1000)
    response: str = Field(..., min_length=10, max_length=500)

    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    tier_used: int = Field(..., ge=1, le=3)

    safe: bool
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    category: Optional[str] = None
    sources: Optional[List[str]] = None
    escalation_required: bool = False

    @validator('response')
    def validate_no_pii(cls, v):
        import re
        patterns = [
            r'\b\d{9,16}\b',
            r'\b[A-Z]{5}\d{4}[A-Z]\b',
            r'(?:INR|Rs\.?)+\s*\d+',
        ]
        for pattern in patterns:
            if re.search(pattern, v):
                raise ValueError("Response contains prohibited content")
        return v

    @validator('confidence')
    def validate_confidence(cls, v, values):
        tier = values.get('tier_used')
        if tier == 1 and v < 0.85:
            raise ValueError("Tier 1 requires confidence >= 0.85")
        return v


class ResponseFormatter:
    """Format responses with Pydantic validation"""

    def format(self, orchestrator_response) -> FormattedResponse:
        import uuid

        return FormattedResponse(
            response_id=str(uuid.uuid4()),
            query=orchestrator_response.query,
            response=orchestrator_response.response,
            intent=orchestrator_response.intent,
            confidence=orchestrator_response.confidence,
            tier_used=orchestrator_response.tier_used,
            safe=orchestrator_response.safe,
            processing_time_ms=orchestrator_response.processing_time_ms,
            category=orchestrator_response.metadata.get('category'),
            sources=orchestrator_response.metadata.get('tier_metadata', {}).get('sources'),
            escalation_required=orchestrator_response.metadata.get('tier_metadata', {}).get('requires_human', False)
        )


__all__ = ['ResponseFormatter', 'FormattedResponse']
