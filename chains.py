"""
Module for creating and configuring LangChain chains.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_basic_chain(llm):
    """
    Create a basic LangChain chain.
    
    Args:
        llm: Language model to use in the chain
        
    Returns:
        A configured LangChain chain
    """
    # Define a prompt template
    prompt_template = """
    You are an AI assistant designed to be helpful, harmless, and honest.

    User Query: {query}
    
    Respond to the user's query with accurate, helpful information:
    """
    
    # Create the prompt template
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query"],
    )
    
    # Create and return the chain
    chain = prompt | llm | StrOutputParser()
    
    return chain
