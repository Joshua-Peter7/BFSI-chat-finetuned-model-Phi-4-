"""Context extraction"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ConversationContext:
    session_id: str
    messages: List[Dict] = field(default_factory=list)
    previous_intent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ContextExtractor:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_history = self.config.get('max_history_length', 5)
        self.timeout_min = self.config.get('session_timeout_minutes', 30)
        self.contexts: Dict[str, ConversationContext] = {}
    
    def extract(self, text: str, session_id: str, additional_context: Dict = None) -> Dict:
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(session_id=session_id)
        
        context = self.contexts[session_id]
        
        # Check expiry
        if datetime.utcnow() - context.last_updated > timedelta(minutes=self.timeout_min):
            context = ConversationContext(session_id=session_id)
            self.contexts[session_id] = context
        
        # Add message
        context.messages.append({
            'text': text,
            'timestamp': datetime.utcnow(),
            'role': 'user'
        })
        
        # Trim history
        if len(context.messages) > self.max_history * 2:
            context.messages = context.messages[-(self.max_history * 2):]
        
        context.last_updated = datetime.utcnow()
        
        return {
            'session_id': session_id,
            'previous_intent': context.previous_intent,
            'message_count': len([m for m in context.messages if m['role'] == 'user'])
        }


__all__ = ['ContextExtractor', 'ConversationContext']
