"""
Test script to verify the GenAI API functionality.
"""
import requests
import json

# Define the API base URL
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint of the API."""
    response = requests.get(f"{BASE_URL}/")
    print("Root endpoint response:", response.json())
    assert response.status_code == 200

def test_chain_endpoint():
    """Test the chain endpoint of the API."""
    data = {
        "query": "What is LangChain and why is it useful?", 
        "temperature": 0.7, 
        "max_tokens": 500
    }
    
    response = requests.post(f"{BASE_URL}/chain", json=data)
    print("\nChain endpoint response:", json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_graph_endpoint():
    """Test the graph endpoint of the API."""
    data = {
        "query": "What is LangGraph and how does it extend LangChain?", 
        "temperature": 0.7, 
        "max_tokens": 500
    }
    
    response = requests.post(f"{BASE_URL}/graph", json=data)
    print("\nGraph endpoint response:", json.dumps(response.json(), indent=2))
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing the GenAI API...")
    
    try:
        test_root_endpoint()
        print("✅ Root endpoint test passed!")
        
        test_chain_endpoint()
        print("✅ Chain endpoint test passed!")
        
        test_graph_endpoint()
        print("✅ Graph endpoint test passed!")
        
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure the API server is running with 'uvicorn main:app --reload'")
