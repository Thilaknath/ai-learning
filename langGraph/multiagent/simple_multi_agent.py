"""
Simple Multi-Agent System - Step by Step Learning
================================================

This is a simplified version to understand multi-agent workflows.
We'll start without external APIs and build up gradually.
"""

import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# =============================================================================
# SIMPLE TOOLS (No external APIs needed)
# =============================================================================

@tool
def simple_calculator(expression: str) -> str:
    """
    Calculate simple math expressions.
    
    Args:
        expression: Math expression like "2+2" or "10*5"
    
    Returns:
        Result of the calculation
    """
    try:
        # Only allow basic math operations for safety
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations allowed"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text for basic statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Basic text statistics
    """
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "character_count": len(text),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    return f"Text Analysis: {stats}"


# =============================================================================
# AGENT STATE
# =============================================================================

class SimpleAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: str
    research_result: str
    analysis_result: str
    final_result: str


# =============================================================================
# AGENT NODES
# =============================================================================

def researcher_node(state: SimpleAgentState):
    """Research agent that actually uses LLM."""
    print("🔍 Research Agent is working...")
    
    # Get user message
    user_message = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    # Create LLM and actually call it
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    research_prompt = f"""You are a Research Agent. Your job is to research the following topic: {user_message}

Please provide detailed research findings. Since you don't have access to real-time data, provide what you know about this topic from your training data and suggest what additional information would be needed."""
    
    response = llm.invoke([SystemMessage(content=research_prompt)])
    research_result = response.content
    
    return {
        "current_step": "research_done",
        "research_result": research_result,
        "messages": [AIMessage(content=f"Research Agent: {research_result}")]
    }


def analyzer_node(state: SimpleAgentState):
    """Analysis agent that actually uses LLM."""
    print("📊 Analysis Agent is working...")
    
    research_data = state.get("research_result", "No research data")
    
    # Create LLM and actually call it
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    analysis_prompt = f"""You are an Analysis Agent. Your job is to analyze the research data provided below:

RESEARCH DATA:
{research_data}

Please provide:
1. Key insights from the research
2. Patterns or trends you notice
3. Potential implications
4. Areas that need more investigation

Be analytical and critical in your assessment."""
    
    response = llm.invoke([SystemMessage(content=analysis_prompt)])
    analysis_result = response.content
    
    return {
        "current_step": "analysis_done", 
        "analysis_result": analysis_result,
        "messages": [AIMessage(content=f"Analysis Agent: {analysis_result}")]
    }


def coordinator_node(state: SimpleAgentState):
    """Coordinator that actually uses LLM for final decisions."""
    print("🎯 Coordinator Agent is working...")
    
    research = state.get("research_result", "No research")
    analysis = state.get("analysis_result", "No analysis")
    
    # Create LLM and actually call it
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    coordinator_prompt = f"""You are a Coordinator Agent. Your job is to synthesize the work from the Research and Analysis agents and provide final recommendations.

RESEARCH FINDINGS:
{research}

ANALYSIS RESULTS:
{analysis}

Please provide:
1. A summary of key findings
2. Final recommendations
3. Next steps
4. Any concerns or limitations

Be decisive and actionable in your recommendations."""
    
    response = llm.invoke([SystemMessage(content=coordinator_prompt)])
    final_result = response.content
    
    return {
        "current_step": "completed",
        "final_result": final_result,
        "messages": [AIMessage(content=f"Coordinator Agent: {final_result}")]
    }


# =============================================================================
# WORKFLOW ROUTING
# =============================================================================

def decide_next_step(state: SimpleAgentState) -> Literal["researcher", "analyzer", "coordinator", "__end__"]:
    """Decide which agent should run next."""
    current_step = state.get("current_step", "start")
    
    if current_step == "start":
        return "researcher"
    elif current_step == "research_done":
        return "analyzer"
    elif current_step == "analysis_done":
        return "coordinator"
    else:
        return "__end__"


# =============================================================================
# CREATE THE GRAPH
# =============================================================================

def create_simple_graph():
    """Create a simple multi-agent graph."""
    
    # Create the workflow
    workflow = StateGraph(SimpleAgentState)
    
    # Add nodes (agents)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("coordinator", coordinator_node)
    
    # Set starting point
    workflow.set_entry_point("researcher")
    
    # Add edges (workflow flow)
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run the simple multi-agent system."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("🚀 Simple Multi-Agent System Starting...")
    print("=" * 50)
    
    # Create the workflow
    app = create_simple_graph()
    
    # Get user input
    user_query = input("Enter your question: ").strip()
    if not user_query:
        user_query = "What is artificial intelligence?"
    
    print(f"\n🔍 Processing: {user_query}")
    print("=" * 50)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "current_step": "start",
        "research_result": "",
        "analysis_result": "",
        "final_result": ""
    }
    
    # Run the workflow
    try:
        step_count = 0
        for output in app.stream(initial_state):
            step_count += 1
            print(f"\n📍 Step {step_count}")
            print("-" * 30)
            
            # Show which agent is active and their output
            if "current_step" in output:
                step = output["current_step"]
                if step == "research_done":
                    print("✅ Research Agent completed")
                    if "research_result" in output:
                        print(f"Research Output:\n{output['research_result']}")
                elif step == "analysis_done":
                    print("✅ Analysis Agent completed")
                    if "analysis_result" in output:
                        print(f"Analysis Output:\n{output['analysis_result']}")
                elif step == "completed":
                    print("✅ Coordinator Agent completed")
                    if "final_result" in output:
                        print(f"Final Output:\n{output['final_result']}")
            
            print()  # Add spacing
        
        print(f"\n✅ Workflow completed in {step_count} steps!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
