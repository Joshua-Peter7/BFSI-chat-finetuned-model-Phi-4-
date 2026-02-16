"""Intent Classification Engine"""

from typing import Dict, List, Tuple
import yaml
import re


class IntentClassifier:
    """Classify user intent using keyword matching"""
    
    def __init__(self, config_path: str = "config/intent_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['intent_engine']['intent_classification']
        self.intent_categories = self.config.get('intent_categories', {})
        self.confidence_threshold = self.config.get('confidence_threshold', 0.70)
        
        # Build keyword mappings
        self.intent_keywords = self._build_keyword_mapping()
    
    def _build_keyword_mapping(self) -> Dict[str, List[str]]:
        """Build intent-to-keywords mapping"""
        keywords = {
            # Loan intents
            'loan_eligibility': ['loan', 'eligible', 'eligibility', 'qualify', 'apply'],
            'loan_application_status': ['loan', 'application', 'status', 'approved', 'pending', 'rejected'],
            'loan_documents': ['loan', 'documents', 'papers', 'required', 'needed'],
            
            # EMI intents
            'emi_details': ['emi', 'amount', 'installment', 'monthly', 'payment'],
            'emi_schedule': ['emi', 'schedule', 'due', 'dates', 'when'],
            'emi_missed': ['emi', 'missed', 'late', 'overdue', 'pending'],
            'emi_bounced': ['emi', 'bounced', 'failed', 'returned', 'bounce'],
            
            # Payment intents
            'payment_failure': ['payment', 'failed', 'failure', 'unsuccessful', 'declined'],
            'transaction_status': ['transaction', 'status', 'payment', 'transfer'],
            'payment_methods': ['payment', 'method', 'options', 'how', 'pay'],
            
            # Account intents
            'account_locked': ['account', 'locked', 'blocked', 'frozen', 'suspended'],
            'account_statement': ['statement', 'account', 'transactions', 'history'],
            'account_balance': ['balance', 'account', 'amount', 'available'],
            
            # Profile intents
            'update_mobile': ['update', 'change', 'mobile', 'number', 'phone'],
            'update_address': ['update', 'change', 'address', 'location'],
            'update_email': ['update', 'change', 'email', 'mail'],
            
            # Insurance / policy intents (plural + variants so "policies" / "bank policies" match)
            'policy_information': ['policy', 'policies', 'insurance', 'information', 'details', 'bank policies', 'terms and conditions'],
            'premium_payment': ['premium', 'payment', 'insurance', 'pay'],
            'claim_status': ['claim', 'status', 'insurance', 'submitted'],
            
            # Escalation intents
            'complaint': ['complaint', 'complain', 'issue', 'problem'],
            'speak_to_manager': ['manager', 'supervisor', 'senior', 'escalate'],
            'not_satisfied': ['not satisfied', 'unhappy', 'disappointed'],
        }
        
        return keywords
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify intent using keyword matching
        
        Returns:
            (intent, confidence_score)
        """
        text_lower = text.lower()
        tokens = set(re.findall(r'\b\w+\b', text_lower))
        
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if matches > 0:
                # Score based on keyword matches
                score = matches / len(keywords)
                intent_scores[intent] = score
        
        if not intent_scores:
            return 'unknown', 0.0
        
        # Get top intent
        top_intent = max(intent_scores, key=intent_scores.get)
        top_score = intent_scores[top_intent]
        return top_intent, top_score
    
    def get_category(self, intent: str) -> str:
        """Get category for an intent"""
        for category, intents in self.intent_categories.items():
            if intent in intents:
                return category
        return 'unknown'


__all__ = ['IntentClassifier']