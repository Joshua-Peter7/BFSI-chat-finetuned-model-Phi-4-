"""Text normalization"""

import re


class TextNormalizer:
    CONTRACTIONS = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "what's": "what is",
    }
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lowercase = self.config.get('lowercase', True)
        self.remove_extra_spaces = self.config.get('remove_extra_spaces', True)
        self.expand_contractions = self.config.get('expand_contractions', True)
    
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.CONTRACTIONS.items():
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text


__all__ = ['TextNormalizer']
