# GenAI API with LangChain and LangGraph

A simple Python API that leverages LangChain and LangGraph for creating GenAI applications.

## Setup

1. Create a virtual environment (already done):
```bash
python -m venv PythonBoilderplateAIV1
```

2. Activate the virtual environment:
```bash
# On Windows
PythonBoilderplateAIV1\Scripts\activate.bat

# On Linux/Mac
source PythonBoilderplateAIV1/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
Edit the `.env` file and add your OpenAI API key.

## Running the API

Start the API server:
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

- `GET /`: Root endpoint that returns a welcome message
- `POST /chain`: Endpoint to query a LangChain chain
- `POST /graph`: Endpoint to query a LangGraph workflow

## Example Requests

### Query the Chain

```bash
curl -X POST "http://localhost:8000/chain" -H "Content-Type: application/json" -d '{"query": "Tell me about artificial intelligence.", "temperature": 0.7, "max_tokens": 500}'
```

### Query the Graph

```bash
curl -X POST "http://localhost:8000/graph" -H "Content-Type: application/json" -d '{"query": "What are the benefits of using LangChain?", "temperature": 0.7, "max_tokens": 500}'
```

## Project Structure

- `main.py`: FastAPI application with API endpoints
- `llm_service.py`: Module for configuring and initializing LLMs
- `chains.py`: Module for creating LangChain chains
- `graphs.py`: Module for creating LangGraph workflows
- `.env`: Environment variables for configuration
- `requirements.txt`: Project dependencies
