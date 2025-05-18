"""
LLM service module to handle the initialization and configuration of LLMs.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


def get_llm(temperature=0.7, max_tokens=1000, model_name=None):
    """
    Initialize and return a properly configured LLM.
    
    Args:
        temperature: Sampling temperature for the LLM
        max_tokens: Maximum number of tokens in the response
        model_name: Optional model name override
        
    Returns:
        A configured LLM instance
    """
    # Get model name and provider from environment variables or use defaults
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "llama3-8b-8192")
    
    llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    print(f"Using model: {model_name} from provider: {llm_provider}")
    
    # Initialize the LLM based on the provider
    if llm_provider == "groq":
        llm = ChatGroq(
            model=model_name,  # Use llama3-8b-8192
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        # Fallback to OpenAI
        llm = ChatOpenAI(
            model='ss',
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    return llm
