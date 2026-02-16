"""Test complete safety layer"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.safety import SafetyLayer


def main():
    print("=" * 60)
    print("Testing Safety and Compliance Layer")
    print("=" * 60)

    safety = SafetyLayer()

    test_cases = [
        {
            "text": "For your EMI details, please log in to our mobile app.",
            "expected": "PASS",
            "description": "Safe compliant response",
        },
        {
            "text": "Your EMI is INR 25000 per month.",
            "expected": "FAIL",
            "description": "Contains specific amount",
        },
        {
            "text": "Your account number is 1234567890.",
            "expected": "FAIL",
            "description": "Contains account number",
        },
        {
            "text": "You should invest all your money in stocks for guaranteed 15% returns.",
            "expected": "FAIL",
            "description": "Financial advice and guaranteed returns",
        },
        {
            "text": "Contact me at john@example.com or 9876543210.",
            "expected": "FAIL",
            "description": "PII leakage",
        },
        {
            "text": "To update your address, please visit our website or mobile app.",
            "expected": "PASS",
            "description": "Safe informational response",
        },
    ]

    print("\n" + "=" * 60)
    print("Running Safety Checks")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print("-" * 60)
        print(f"Text: {test['text']}")

        result = safety.check(test['text'])

        actual = "PASS" if result.is_safe else "FAIL"
        expected = test['expected']

        print(f"\nExpected: {expected}")
        print(f"Actual: {actual}")

        if actual == expected:
            print("CORRECT")
            passed += 1
        else:
            print("INCORRECT")
            failed += 1

        if not result.is_safe:
            print("\nViolations:")
            if not result.safety_result.is_safe:
                print(f"  Safety: {', '.join(result.safety_result.violations)}")
            if not result.compliance_result.is_compliant:
                for v in result.compliance_result.violations:
                    print(f"  Compliance: {v['message']}")

            print("\nFallback response:")
            print(f"  {result.final_response}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(test_cases)*100:.1f}%")

    if failed == 0:
        print("\nAll tests passed. Safety layer working correctly.")
    else:
        print("\nSome tests failed. Review safety configuration.")


if __name__ == "__main__":
    main()
