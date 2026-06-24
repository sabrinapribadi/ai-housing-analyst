import sys
sys.path.append('.')
from src.agent.agent import AirbnbAgent

print("🤖 Initializing Airbnb Agent...")
agent = AirbnbAgent()

questions = [
    "What's the average price of an Airbnb in Tokyo?",
    "Which neighborhoods have the highest average prices?",
    "Show me the price distribution",
    "What are the cheapest neighborhoods?"
]

for q in questions:
    print("\n" + "=" * 60)
    print(f"❓ Question: {q}")
    print("=" * 60)
    response = agent.ask(q)
    print(f"🤖 Response:\n{response}")
