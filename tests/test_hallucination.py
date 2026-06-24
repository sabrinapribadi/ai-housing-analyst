"""
Hallucination detection tests for the LLM agent.
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.agent.agent import AirbnbAgent


class HallucinationDetector:
    """Detect hallucinations in agent responses."""

    def __init__(self):
        self.agent = AirbnbAgent()
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> dict:
        """Load known facts from the data."""
        df = pd.read_parquet("data/processed/listings_processed.parquet")
        return {
            "total_listings": len(df),
            "avg_price": df["price"].mean(),
            "median_price": df["price"].median(),
            "top_neighborhood": df.groupby("neighbourhood_cleansed")["price"].mean().idxmax(),
            "top_neighborhood_price": df.groupby("neighbourhood_cleansed")["price"].mean().max(),
            "cheapest_neighborhood": df.groupby("neighbourhood_cleansed")["price"].mean().idxmin(),
            "cheapest_neighborhood_price": df.groupby("neighbourhood_cleansed")["price"].mean().min(),
            "room_type_counts": df["room_type"].value_counts().to_dict(),
            "neighborhoods": df["neighbourhood_cleansed"].unique().tolist(),
        }

    def extract_numbers_from_response(self, response: str) -> list:
        """Extract all numbers from the response."""
        numbers = re.findall(r"[¥]?[\d,]+\.?\d*", response)
        cleaned = []
        for num in numbers:
            cleaned_num = num.replace("¥", "").replace(",", "")
            try:
                cleaned.append(float(cleaned_num))
            except ValueError:
                pass
        return cleaned

    def detect_price_errors(self, question: str, response: str) -> dict:
        """Check if reported prices match ground truth."""
        errors = []
        prices = self.extract_numbers_from_response(response)

        if "average price" in question.lower() or "avg price" in question.lower():
            if prices:
                reported_price = prices[0]
                true_price = self.ground_truth["avg_price"]
                diff_pct = abs(reported_price - true_price) / true_price * 100

                if diff_pct > 10:
                    errors.append({
                        "type": "price_mismatch",
                        "reported": reported_price,
                        "true": true_price,
                        "diff_pct": diff_pct,
                        "severity": "high" if diff_pct > 20 else "medium",
                    })

        return {"has_errors": len(errors) > 0, "errors": errors}

    def detect_fabricated_facts(self, response: str) -> dict:
        """Detect if the response fabricates data."""
        errors = []

        if "exactly" in response.lower() or "precisely" in response.lower():
            errors.append({
                "type": "overconfidence",
                "message": "Response uses absolute language",
                "severity": "low",
            })

        if "no data" in response.lower() or "not available" in response.lower():
            if "reviews" in response.lower():
                reviews_path = "data/processed/reviews_processed.parquet"
                if os.path.exists(reviews_path):
                    df = pd.read_parquet(reviews_path)
                    if len(df) > 0:
                        errors.append({
                            "type": "false_negative",
                            "message": "Claimed no review data when it exists",
                            "severity": "high",
                        })

        return {"has_errors": len(errors) > 0, "errors": errors}

    def test_question(self, question: str) -> dict:
        """Test a question for hallucination."""
        print(f"\n🔍 Testing: {question}")

        response = self.agent.ask(question)
        print(f"   Response: {response[:200]}...")

        price_checks = self.detect_price_errors(question, response)
        fabrication_checks = self.detect_fabricated_facts(response)

        all_errors = price_checks["errors"] + fabrication_checks["errors"]
        has_hallucination = len(all_errors) > 0

        return {
            "question": question,
            "response": response,
            "has_hallucination": has_hallucination,
            "errors": all_errors,
            "severity": (
                "high" if any(e.get("severity") == "high" for e in all_errors)
                else "medium" if all_errors
                else "none"
            ),
        }


def run_hallucination_tests():
    """Run a suite of hallucination tests."""
    detector = HallucinationDetector()

    test_questions = [
        "What's the average price of an Airbnb in Tokyo?",
        "Which neighborhood has the highest average prices?",
        "How many listings are there in Tokyo?",
        "What's the cheapest neighborhood?",
        "Show me the price distribution",
        "What's the most popular room type?",
    ]

    results = []
    print("=" * 60)
    print("HALLUCINATION DETECTION TEST SUITE")
    print("=" * 60)

    for question in test_questions:
        result = detector.test_question(question)
        results.append(result)

        status = "PASS" if not result["has_hallucination"] else "FAIL"
        print(f"\n   Status: {status}")
        for error in result["errors"]:
            print(
                f"   [{error['type']}] "
                f"{error.get('message', '')} "
                f"(severity: {error.get('severity', 'unknown')})"
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for r in results if not r["has_hallucination"])
    print(f"Tests:                  {total}")
    print(f"Passed:                 {passed} ({passed / total * 100:.1f}%)")
    print(f"Hallucinations detected:{total - passed}")

    return results


if __name__ == "__main__":
    run_hallucination_tests()
