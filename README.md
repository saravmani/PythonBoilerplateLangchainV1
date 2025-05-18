# GenAI API with LangChain and LangGraph

A simple Python API that leverages LangChain and LangGraph for creating GenAI applications with Groq's Llama3-8b-8192 model.

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
Edit the `.env` file and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
You can obtain a Groq API key from https://console.groq.com/keys

## Running the API

Start the API server:
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## Testing the Groq LLM

To verify that your Groq LLM integration is working correctly, run the test script:

```bash
python test_groq.py
```

This script will send a test prompt to the Llama3-8b-8192 model through the Groq API and display the response.

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
