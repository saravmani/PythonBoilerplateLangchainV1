"""
Test script to verify Groq LLM functionality.
"""
import os
from dotenv import load_dotenv
from llm_service import get_llm

def test_groq_llm():
    """Test that the Groq LLM with Llama3-8b-8192 is working correctly."""
    # Load environment variables
    load_dotenv()
    
    # Set up environment for Groq
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY not found in .env file.")
        print("Please add your Groq API key to the .env file.")
        print("You can get an API key from https://console.groq.com/keys")
        return
    
    print("Testing Groq with Llama3-8b-8192...")
    
    # Create the Groq LLM
    llm = get_llm(temperature=0.7, max_tokens=500)
    
    # Test a simple prompt
    prompt = "Explain the benefits of using Llama3 models in 3 sentences:"
    print(f"\nPrompt: {prompt}")
    
    try:
        # Generate a response
        response = llm.invoke(prompt)
        
        # Print the response
        if hasattr(response, "content"):
            print(f"\nResponse: {response.content}")
        else:
            print(f"\nResponse: {response}")
            
        print("\n✅ Groq LLM test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error testing Groq LLM: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    test_groq_llm()
