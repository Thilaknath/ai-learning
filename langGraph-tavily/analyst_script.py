import os
import sys
import argparse
from typing import TypedDict, Literal
# NEW: Import for verbose logging
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# --- 1. Define the State for our Graph ---
class AgentState(TypedDict):
    error_context: str
    key_error: str
    search_results: str
    analysis: str
    decision: Literal["continue", "retry"]
    retry_count: int

# --- 2. Define the Tools & Nodes ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(max_results=3)

def mock_search_node(state: AgentState):
    """Mocks the search for solutions to the identified key error."""
    print("---NODE: MOCK SEARCHING FOR SOLUTION---", file=sys.stderr)
    # Mock search results
    mock_results = [
        {"content": "Mock result 1 for key error: " + state["key_error"]},
        {"content": "Mock result 2 for key error: " + state["key_error"]},
        {"content": "Mock result 3 for key error: " + state["key_error"]}
    ]
    search_results = "\n".join([res["content"] for res in mock_results])
    return {"search_results": search_results}

def identify_error_node(state: AgentState):
    """Identifies the key error message from the context."""
    print("---NODE: IDENTIFYING ERROR---", file=sys.stderr)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert DevOps engineer. Read the following error log and identify the single, most critical error message that caused the build to fail. Output only that one error message and nothing else.\n\nLOG:\n{error_context}"
    )
    chain = prompt | llm
    key_error = chain.invoke({"error_context": state["error_context"]}).content
    return {"key_error": key_error}

def search_node(state: AgentState):
    """Searches for solutions to the identified key error."""
    print("---NODE: SEARCHING FOR SOLUTION---", file=sys.stderr)
    results = search_tool.invoke(state["key_error"])
    search_results = "\n".join([res["content"] for res in results])
    return {"search_results": search_results}

def reflection_node(state: AgentState):
    """Reflects on the search results and decides whether to continue or retry, with retry limit."""
    print("---NODE: REFLECTING ON SEARCH RESULTS---", file=sys.stderr)
    retry_limit = 3
    retry_count = state.get("retry_count", 0)
    if retry_count >= retry_limit:
        print(f"[INFO] Retry limit reached ({retry_limit}), forcing 'continue'", file=sys.stderr)
        return {"decision": "continue", "retry_count": retry_count}
    prompt = ChatPromptTemplate.from_template(
        """You are an expert troubleshooter. You have identified a key error and received some search results.
        Evaluate whether the search results are relevant and sufficient to solve the problem.
        If the results seem helpful, respond with 'continue'.
        If the results are irrelevant or insufficient, respond with 'retry'.

        Key Error: {key_error}
        Search Results: {search_results}

        Decision ('continue' or 'retry'):"""
    )
    chain = prompt | llm
    decision = chain.invoke(state).content
    if decision.strip() == "retry":
        retry_count += 1
    else:
        retry_count = state.get("retry_count", 0)
    return {"decision": decision.strip(), "retry_count": retry_count}

def synthesize_analysis_node(state: AgentState):
    """Generates the final analysis report."""
    print("---NODE: SYNTHESIZING ANALYSIS---", file=sys.stderr)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful DevOps assistant. Your job is to create a clear, concise failure analysis report in Markdown format.

        **Original Error Context:**
        {error_context}

        **Key Error Identified:**
        {key_error}
        
        **Relevant Information from Web Search:**
        {search_results}

        **Your Task:**
        Based on all the information above, create a report with the following sections:
        1.  **Summary:** A one-sentence summary of the problem.
        2.  **Probable Cause:** A brief explanation of why this error likely occurred.
        3.  **Suggested Fix:** A clear, actionable suggestion for how to fix the problem.
        """
    )
    chain = prompt | llm
    analysis = chain.invoke(state).content
    return {"analysis": analysis}

# --- 3. Build the Graph with Conditional Logic ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("identify_error", identify_error_node)
graph_builder.add_node("search", mock_search_node)
graph_builder.add_node("reflect", reflection_node)
graph_builder.add_node("synthesize", synthesize_analysis_node)

graph_builder.set_entry_point("identify_error")
graph_builder.add_edge("identify_error", "search")
graph_builder.add_edge("search", "reflect")
graph_builder.add_conditional_edges(
    "reflect",
    lambda state: state["decision"],
    {
        "retry": "search",
        "continue": "synthesize"
    }
)
graph_builder.add_edge("synthesize", END)

app = graph_builder.compile()

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # NEW: Enable verbose logging
    # set_debug(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, help="The error log or context to analyze.")
    args = parser.parse_args()

    inputs = {"error_context": args.context, "retry_count": 0}

    final_state = app.invoke(inputs)
    # Only print the final report to stdout
    print(final_state['analysis'])

    # Write the analysis to report.md for CI workflow compatibility
    with open("report.md", "w") as f:
        f.write(final_state['analysis'])
