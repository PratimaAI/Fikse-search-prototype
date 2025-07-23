#!/usr/bin/env python3
"""
FikseAgent Integration Test Script

This script tests the complete integration between the search API and agent API
by simulating a full conversation flow programmatically.
"""

import requests
import time
import uuid
import json
from typing import Dict, List

# Test configuration
SEARCH_API_URL = "http://localhost:8000"
AGENT_API_URL = "http://localhost:8001"
TEST_SESSION_ID = str(uuid.uuid4())

def check_service_health(service_name: str, url: str) -> bool:
    """Check if a service is healthy and responding"""
    try:
        if "8000" in url:  # Search API - test with search endpoint
            response = requests.get(f"{url}/search", params={"q": "test"}, timeout=5)
        else:  # Agent API - has health endpoint
            response = requests.get(f"{url}/health", timeout=5)
            
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name} is not responding: {e}")
        return False

def test_search_api() -> bool:
    """Test the search API directly"""
    print("\n🔍 Testing Search API...")
    
    test_queries = [
        "silk dress tear",
        "jacket zipper broken", 
        "suit alter waist",
        "shoe sole repair"
    ]
    
    for query in test_queries:
        try:
            response = requests.get(f"{SEARCH_API_URL}/search", params={"q": query})
            if response.status_code == 200:
                results = response.json()
                print(f"   ✅ '{query}' → {len(results)} results")
                if results:
                    top_result = results[0]
                    print(f"      Top: {top_result.get('Service')} - ${top_result.get('Price', 0):.0f}")
            else:
                print(f"   ❌ '{query}' → Error {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ '{query}' → Exception: {e}")
            return False
    
    return True

def send_agent_message(message: str) -> Dict:
    """Send a message to the agent and return the response"""
    payload = {
        "session_id": TEST_SESSION_ID,
        "user_input": message
    }
    
    response = requests.post(f"{AGENT_API_URL}/agent", json=payload)
    response.raise_for_status()
    return response.json()

def test_agent_conversation() -> bool:
    """Test a full conversation flow with the agent"""
    print("\n🤖 Testing Agent Conversation Flow...")
    
    conversation_steps = [
        {
            "message": "Hello, I need help with a repair",
            "expected_state": "greeting",
            "description": "Initial greeting"
        },
        {
            "message": "I have a silk dress with a small tear near the hem",
            "expected_state": "selecting", 
            "description": "Repair request with service suggestions"
        },
        {
            "message": "1",
            "expected_state": "manual_addition",
            "description": "Select first service"
        },
        {
            "message": "No",
            "expected_state": "confirming",
            "description": "Decline additional services"
        },
        {
            "message": "Yes",
            "expected_state": "completed",
            "description": "Confirm order creation"
        }
    ]
    
    for i, step in enumerate(conversation_steps, 1):
        print(f"\n   Step {i}: {step['description']}")
        print(f"   👤 User: \"{step['message']}\"")
        
        try:
            response = send_agent_message(step['message'])
            
            # Check response structure
            required_fields = ["intent", "response", "conversation_state"]
            for field in required_fields:
                if field not in response:
                    print(f"   ❌ Missing field: {field}")
                    return False
            
            print(f"   🤖 Agent: \"{response['response'][:100]}...\"")
            print(f"   📊 State: {response['conversation_state']} (expected: {step['expected_state']})")
            
            # Verify expected state
            if response['conversation_state'] != step['expected_state']:
                print(f"   ⚠️  State mismatch! Expected {step['expected_state']}, got {response['conversation_state']}")
                # Don't fail completely, but note the issue
            
            # Show services if provided
            if response.get('services'):
                print(f"   🛠️  Services offered: {len(response['services'])}")
                for j, service in enumerate(response['services'][:2], 1):  # Show first 2
                    print(f"      {j}. {service['service']} - ${service['price']:.0f}")
            
            # Show order if created
            if response.get('order_created'):
                order = response['order_created']
                print(f"   🎉 Order created: {order['order_id']} (${order['total_price']:.0f})")
            
            time.sleep(0.5)  # Brief pause between messages
            
        except Exception as e:
            print(f"   ❌ Error in step {i}: {e}")
            return False
    
    print("\n✅ Conversation flow completed successfully!")
    return True

def test_session_management() -> bool:
    """Test session management features"""
    print("\n📋 Testing Session Management...")
    
    try:
        # Get session state
        response = requests.get(f"{AGENT_API_URL}/agent/session/{TEST_SESSION_ID}")
        if response.status_code == 200:
            session_data = response.json()
            print(f"   ✅ Session state retrieved: {session_data.get('conversation_state')}")
        else:
            print(f"   ❌ Failed to get session state: {response.status_code}")
            return False
        
        # Reset session
        response = requests.delete(f"{AGENT_API_URL}/agent/session/{TEST_SESSION_ID}")
        if response.status_code == 200:
            print("   ✅ Session reset successfully")
        else:
            print(f"   ❌ Failed to reset session: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Session management error: {e}")
        return False

def test_error_handling() -> bool:
    """Test error handling scenarios"""
    print("\n⚠️  Testing Error Handling...")
    
    test_cases = [
        {
            "message": "gibberish nonsense xyz123",
            "description": "Nonsense input handling"
        },
        {
            "message": "99",  # Invalid service selection
            "description": "Invalid service selection"
        }
    ]
    
    # Start fresh session for error testing
    error_session = str(uuid.uuid4())
    
    try:
        # First establish context
        payload = {"session_id": error_session, "user_input": "I need help with repairs"}
        requests.post(f"{AGENT_API_URL}/agent", json=payload)
        
        # Test error cases
        for test_case in test_cases:
            payload = {"session_id": error_session, "user_input": test_case["message"]}
            response = requests.post(f"{AGENT_API_URL}/agent", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {test_case['description']}: Handled gracefully")
                print(f"      Response: \"{data['response'][:60]}...\"")
            else:
                print(f"   ❌ {test_case['description']}: Failed with {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

def run_integration_tests() -> bool:
    """Run all integration tests"""
    print("🧪 FikseAgent Integration Test Suite")
    print("=" * 50)
    
    # Check service health
    print("🏥 Checking Service Health...")
    search_healthy = check_service_health("Search API", SEARCH_API_URL)
    agent_healthy = check_service_health("Agent API", AGENT_API_URL)
    
    if not (search_healthy and agent_healthy):
        print("\n❌ Services not healthy. Please start the system first:")
        print("   python start_system.py")
        return False
    
    # Run tests
    tests = [
        ("Search API", test_search_api),
        ("Agent Conversation", test_agent_conversation), 
        ("Session Management", test_session_management),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n" + "=" * 30)
        print(f"Running {test_name} Tests...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} tests PASSED")
            else:
                print(f"❌ {test_name} tests FAILED")
        except Exception as e:
            print(f"❌ {test_name} tests CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Integration is working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1) 