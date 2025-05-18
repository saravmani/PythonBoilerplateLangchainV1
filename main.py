"""
Main FastAPI application for the GenAI API.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our custom modules
from llm_service import get_llm
from chains import create_basic_chain
from graphs import create_basic_graph

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GenAI API",
    description="A simple API for interacting with LangChain and LangGraph",
    version="1.0.0",
)

# Define request models
class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 1000
    temperature: float = 0.7


class GraphQueryRequest(BaseModel):
    query: str
    max_tokens: int = 1000
    temperature: float = 0.7


@app.get("/")
def read_root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the GenAI API using LangChain and LangGraph!"}


@app.post("/chain")
async def query_chain(request: QueryRequest):
    """
    Endpoint to query a LangChain chain.
    
    Args:
        request: QueryRequest with the user query and parameters
        
    Returns:
        The response from the LangChain chain
    """
    try:
        # Get the LLM
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)
        
        # Create and run the chain
        chain = create_basic_chain(llm)
        result = chain.invoke({"query": request.query})
        
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chain: {str(e)}")


@app.post("/graph")
async def query_graph(request: GraphQueryRequest):
    """
    Endpoint to query a LangGraph graph.
    
    Args:
        request: GraphQueryRequest with the user query and parameters
        
    Returns:
        The response from the LangGraph execution
    """
    try:
        # Get the LLM
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)
        
        # Create and run the graph
        graph = create_basic_graph(llm)
        result = graph.invoke({"query": request.query})
        
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing graph: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
