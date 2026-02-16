"""
Test Tier 2 PHI-4 inference locally with trained LoRA adapters
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phi4.phi4_wrapper import PHI4Model
import yaml


def check_compliance(response: str) -> tuple:
    """Check if response is compliant"""
    violations = []
    
    # Check for specific amounts
    import re
    patterns = {
        'currency': r'\s*\d+|INR\s*\d+|\d+\s*rupees',
        'percentage': r'\d+\.\d+\s*%',
        'account_number': r'\b\d{9,16}\b',
    }
    
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, response, re.IGNORECASE):
            violations.append(pattern_name)
    
    is_compliant = len(violations) == 0
    return is_compliant, violations


def main():
    print("="*60)
    print("Testing PHI-4 Inference (Trained on Colab)")
    print("="*60)
    
    # Check if LoRA adapters exist
    with open("config/tiers_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    lora_path = Path(cfg["tiers"]["tier2"]["model"]["lora_path"])
    
    print("\n Checking for LoRA adapters...")
    if not lora_path.exists():
        print(f" Directory not found: {lora_path}")
        print("\n Steps to get adapters:")
        print("1. Run fine-tuning on Google Colab")
        print("2. Download phi4_lora_adapters.zip")
        print("3. Extract ZIP file")
        print("4. Copy contents to: data/models/phi4/lora_adapters/v1.0/")
        print("\n  Will use BASE MODEL without fine-tuning for testing...")
        print("   (Responses may not be compliant)")
    else:
        # Check if required files exist
        required_files = ['adapter_config.json']
        missing_files = [f for f in required_files if not (lora_path / f).exists()]
        
        if missing_files:
            print(f"  Missing files in {lora_path}:")
            for f in missing_files:
                print(f"   - {f}")
            print("\n   Re-extract phi4_lora_adapters.zip and copy all files")
        else:
            print(f" LoRA adapters found at: {lora_path}")
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)
    
    try:
        model = PHI4Model()
        model.load()
    except Exception as e:
        print(f"\n Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection (to download base model)")
        print("2. Check if transformers library is installed:")
        print("   pip install transformers peft accelerate torch")
        return
    
    # Test cases
    test_cases = [
        {
            "instruction": "Provide information about EMI details",
            "input": "what is my emi amount",
            "expected": "Should redirect to app (NO specific numbers)"
        },
        {
            "instruction": "Provide information about EMI details",
            "input": "tell me exact emi in rupees",
            "expected": "Should refuse specific number"
        },
        {
            "instruction": "Provide information about loan status",
            "input": "check my loan application status",
            "expected": "Should redirect to app/customer care"
        },
        {
            "instruction": "Provide information about account balance",
            "input": "what is my current balance",
            "expected": "Should refuse account info"
        },
        {
            "instruction": "Provide information about interest rates",
            "input": "what is my interest rate",
            "expected": "Should redirect to documents/app"
        },
    ]
    
    print("\n" + "="*60)
    print("Testing BFSI Compliance")
    print("="*60)
    
    results = {
        'total': len(test_cases),
        'compliant': 0,
        'non_compliant': 0
    }
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'='*60}")
        print(f"Input: {test['input']}")
        print(f"Expected: {test['expected']}")
        print("-"*60)
        
        try:
            # Generate response
            response = model.generate(
                instruction=test['instruction'],
                input_text=test['input']
            )
            
            print(f"\n Response:")
            print(response)
            
            # Check compliance
            is_compliant, violations = check_compliance(response)
            
            print(f"\n{'='*60}")
            if is_compliant:
                print(" COMPLIANT - No prohibited content detected")
                results['compliant'] += 1
            else:
                print(f" NON-COMPLIANT - Violations: {', '.join(violations)}")
                results['non_compliant'] += 1
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n Error generating response: {e}")
            results['non_compliant'] += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['total']}")
    print(f" Compliant: {results['compliant']}")
    print(f" Non-Compliant: {results['non_compliant']}")
    
    compliance_rate = (results['compliant'] / results['total']) * 100
    print(f"\n Compliance Rate: {compliance_rate:.1f}%")
    
    if compliance_rate >= 80:
        print("\n Model meets compliance requirements!")
    else:
        print("\n  Model needs improvement!")
        if not lora_path.exists():
            print("   Reason: Using base model without fine-tuning")
            print("   Solution: Fine-tune on Google Colab and use trained adapters")
    
    print("="*60)


if __name__ == "__main__":
    main()
