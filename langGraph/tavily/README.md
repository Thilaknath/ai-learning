# DevOps Error Analysis Agent

A LangGraph-based AI agent that analyzes DevOps error logs, identifies key issues, searches for solutions using Tavily, and generates comprehensive failure analysis reports.

## Features

- **Error Identification**: Automatically extracts the most critical error from build logs
- **Web Search Integration**: Uses Tavily search to find relevant solutions
- **Intelligent Retry Logic**: Reflects on search results and retries if needed (max 3 attempts)
- **Markdown Reports**: Generates structured analysis reports with summary, cause, and fix suggestions

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd langGraph-tavily
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain langchain-openai langchain-community langgraph tavily-python
   ```

3. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export TAVILY_API_KEY="your-tavily-api-key"
   ```

## Usage

Run the script with an error context:

```bash
python analyst_script.py --context "Your error log content here"
```

### Example

```bash
python analyst_script.py --context "ERROR: Docker build failed - unable to locate package nodejs"
```

## Output

- **Console**: Displays the final analysis report
- **File**: Saves the report as `report.md` for CI/CD integration

The report includes:
- **Summary**: One-sentence problem description
- **Probable Cause**: Explanation of why the error occurred
- **Suggested Fix**: Actionable solution steps

## Architecture

The agent uses a LangGraph state machine with four nodes:
1. `identify_error` - Extracts key error from logs
2. `search` - Searches for solutions using Tavily
3. `reflect` - Evaluates search results quality
4. `synthesize` - Generates final analysis report

## Development

Enable debug mode by uncommenting the debug line in the script:
```python
set_debug(True)
```
