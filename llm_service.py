"""
LLM service module to handle the initialization and configuration of LLMs.
"""
import os
from langchain_openai import ChatOpenAI


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
    # Get model name from environment variable or use default
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    print(f"Using model hahaha: {model_name}")
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return llm
