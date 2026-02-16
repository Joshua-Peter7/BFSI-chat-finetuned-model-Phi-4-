"""Tier 2: Fine-tuned Small Language Model (Phi-3 + LoRA).

Uses same instruction format as training (prepare_training_data.py) so model
behavior matches fine-tuning. Refuses guessing; redirects for account-specific data.
"""

from .base_tier import BaseTier, TierResponse
from src.models.phi4.phi4_wrapper import PHI4Model
from typing import Dict, List
import time
import yaml

# Intent → instruction mapping (must match training_data.json instruction phrasing)
INTENT_TO_INSTRUCTION = {
    "loan_eligibility": "Provide information about loan eligibility criteria",
    "loan_application_status": "Check loan application status",
    "loan_documents": "Provide information about loan documents required",
    "emi_details": "Provide EMI details",
    "emi_schedule": "Provide EMI schedule information",
    "emi_missed": "Provide information about missed EMI",
    "emi_bounced": "Provide information about bounced EMI",
    "payment_failure": "Provide information about payment failure",
    "transaction_status": "Check transaction status",
    "payment_methods": "Provide information about payment methods",
    "account_locked": "Provide information about account locked",
    "account_statement": "Provide account statement information",
    "account_balance": "Provide information about account balance",
    "update_mobile": "Provide information about updating mobile number",
    "update_address": "Provide information about updating address",
    "update_email": "Provide information about updating email",
    "policy_information": "Provide information about policy and terms",
    "premium_payment": "Provide information about premium payment",
    "claim_status": "Check insurance claim status",
    "complaint": "Handle complaint",
    "speak_to_manager": "Handle request to speak to manager",
    "not_satisfied": "Handle customer not satisfied",
}


class Tier2SLM(BaseTier):
    """
    Tier 2: Fine-tuned Phi-3 + LoRA (BFSI).
    Deterministic, no guessing; redirects for account-specific details.
    """

    def __init__(self, config_path: str = "config/tiers_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(config['tiers']['tier2'])
        self.tier_number = 2
        self.tier_name = self.config['name']
        self.model = PHI4Model(config_path)

    def _instruction_for(self, intent: str, similar_queries: List[Dict]) -> str:
        """Use KB instruction when available so prompt matches training distribution."""
        if similar_queries:
            meta = similar_queries[0].get("metadata") or {}
            instr = meta.get("instruction", "").strip()
            if instr:
                return instr
        return INTENT_TO_INSTRUCTION.get(intent, f"Provide information about {intent.replace('_', ' ')}")

    def generate(
        self,
        query: str,
        intent: str,
        similar_queries: List[Dict]
    ) -> TierResponse:
        """Generate using fine-tuned model; instruction format matches training."""
        start_time = time.time()
        instruction = self._instruction_for(intent, similar_queries or [])
        try:
            response_text = self.model.generate(instruction=instruction, input_text=query)
        except Exception:
            from src.models.phi4.phi4_wrapper import BFSI_REDIRECT
            response_text = BFSI_REDIRECT
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Calculate confidence based on generation
        # (In production, you'd have a more sophisticated method)
        confidence = 0.75  # Default medium confidence for Tier 2
        
        return TierResponse(
            tier=2,
            text=response_text,
            confidence=confidence,
            generation_time_ms=generation_time_ms,
            source="tier2_phi4",
            metadata={
                'intent': intent,
                'model': 'phi4_finetuned'
            }
        )


__all__ = ['Tier2SLM']