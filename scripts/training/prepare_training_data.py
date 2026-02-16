"""
Prepare BFSI dataset for PHI-4 fine-tuning
Converts Alpaca format to instruction-tuned format
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def format_instruction(instruction, input_text, output):
    """
    Format as instruction-following prompt
    
    Format:
    <|user|>
    {instruction}
    Input: {input_text}
    <|assistant|>
    {output}
    """
    prompt = f"""<|user|>
{instruction}
Input: {input_text}
<|assistant|>
{output}"""
    
    return prompt


def add_refusal_examples(examples):
    """
    Add explicit refusal examples for compliance
    
    CRITICAL: Train model to refuse specific numbers
    """
    refusal_examples = [
        {
            "instruction": "Provide information about EMI details",
            "input": "tell me my exact emi amount in rupees",
            "output": "I cannot provide specific EMI amounts. For your exact EMI details, please log in to our mobile app or internet banking portal where you can view your complete payment schedule."
        },
        {
            "instruction": "Provide information about account balance",
            "input": "what is my current account balance",
            "output": "I cannot access your account balance information. Please log in to our mobile app or internet banking to check your current balance securely."
        },
        {
            "instruction": "Provide information about interest rates",
            "input": "what is my loan interest rate percentage",
            "output": "I cannot provide specific interest rate information. Please refer to your loan agreement or check the mobile app for your interest rate details. You can also contact customer care for assistance."
        },
        {
            "instruction": "Provide information about loan eligibility",
            "input": "am i eligible for a loan",
            "output": "Loan eligibility depends on various factors including your credit score, income, and existing obligations. For a personalized eligibility assessment, please visit our website or contact our loan department."
        },
        {
            "instruction": "Handle requests for account numbers",
            "input": "what is my account number",
            "output": "I cannot provide account numbers for security reasons. You can find your account number in your account statement, passbook, or by logging into internet banking."
        }
    ]
    
    return examples + refusal_examples


def validate_training_example(example):
    """
    Validate that training example is compliant
    
    CRITICAL: No examples should contain specific numbers in output
    """
    output = example['output']
    
    # Check for specific amounts
    import re
    prohibited_patterns = [
        r'₹\s*\d+',           # ₹25000
        r'INR\s*\d+',         # INR 25000
        r'\d+\s*rupees',      # 25000 rupees
        r'\d+\.\d+\s*%',      # 8.5%
        r'EMI.*₹?\d+',        # EMI is ₹5000
        r'balance.*₹?\d+',    # balance is 50000
    ]
    
    for pattern in prohibited_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            print(f"⚠️  WARNING: Output contains prohibited pattern: {pattern}")
            print(f"   Output: {output[:100]}...")
            return False
    
    return True


def prepare_dataset(input_path, output_path):
    """
    Prepare dataset for fine-tuning
    """
    print("="*60)
    print("Preparing Training Data for PHI-4 Fine-tuning")
    print("="*60)
    
    # Load dataset
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"\n✅ Loaded {len(dataset)} examples")
    
    # Add refusal examples
    dataset = add_refusal_examples(dataset)
    print(f"✅ Added refusal examples, total: {len(dataset)}")
    
    # Validate and format
    formatted_data = []
    invalid_count = 0
    
    for example in dataset:
        # Validate
        if not validate_training_example(example):
            invalid_count += 1
            continue
        
        # Format
        formatted_text = format_instruction(
            instruction=example['instruction'],
            input_text=example['input'],
            output=example['output']
        )
        
        formatted_data.append({
            'text': formatted_text
        })
    
    print(f"\n✅ Formatted {len(formatted_data)} valid examples")
    if invalid_count > 0:
        print(f"⚠️  Skipped {invalid_count} invalid examples")
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to: {output_path}")
    
    # Print sample
    print("\n" + "="*60)
    print("Sample Training Example:")
    print("="*60)
    print(formatted_data[0]['text'])
    print("="*60)
    
    return formatted_data


def main():
    input_path = Path("data/raw/bfsi_dataset_alpaca.json")
    output_path = Path("data/processed/training_data.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"❌ Dataset not found: {input_path}")
        return
    
    prepare_dataset(input_path, output_path)


if __name__ == "__main__":
    main()