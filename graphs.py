"""
Module for creating and configuring LangGraph graphs.
"""
from typing import Annotated, TypedDict, Literal

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
# ToolExecutor was moved from langgraph.prebuilt in newer versions
from langgraph.graph import StateGraph


class GraphState(TypedDict):
    """State for the graph execution."""
    query: str
    intermediate_steps: list
    response: str


def create_basic_graph(llm):
    """
    Create a basic LangGraph workflow.
    
    Args:
        llm: Language model to use in the graph
        
    Returns:
        A configured LangGraph workflow
    """
    # Define a state graph
    workflow = StateGraph(GraphState)
    
    # Define the processing function
    def process_query(state):
        """Process the user query and generate a response."""
        query = state["query"]
        
        # Create a prompt template
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
        
        # Generate the response
        formatted_prompt = prompt.format(query=query)
        response = llm.invoke(formatted_prompt)
        
        # Extract the response content
        if hasattr(response, "content"):
            response_content = response.content
        else:
            response_content = str(response)
        
        # Update the state
        return {
            "response": response_content,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Processed query"]
        }
    
    # Add nodes to the graph
    workflow.add_node("process", process_query)
    
    # Set the entry point
    workflow.set_entry_point("process")
    
    # Set the exit point
    workflow.set_finish_point("process")
    
    # Compile the graph
    return workflow.compile()
