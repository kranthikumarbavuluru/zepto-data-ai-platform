"""
Test script for Zepto Support Assistant
Verifying LangGraph routing: Policy Question vs. General Question
"""
import json
from support_assistant.main import QueryRequest, ask_policy

print("=== TEST 1: Policy Question (Triggers Retrieval) ===")
req1 = QueryRequest(query="What is your return and refund policy for damaged groceries?")
resp1 = ask_policy(req1)
print(json.dumps(resp1.model_dump(), indent=2))

print("=== TEST 2: General Question (Direct Canned Answer) ===")
req2 = QueryRequest(query="What is the capital of France?")
resp2 = ask_policy(req2)
print(json.dumps(resp2.model_dump(), indent=2))
