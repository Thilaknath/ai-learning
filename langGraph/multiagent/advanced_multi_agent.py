"""
Advanced Multi-Agent System with LangGraph
==========================================

This system demonstrates:
- Multiple specialized agents working together
- Decision-making and critiquing workflows
- Various tools beyond just web search
- Complex state management and routing
- Agent collaboration and handoffs

Agents:
1. Research Agent - Gathers information from multiple sources
2. Analysis Agent - Processes and analyzes data
3. Critic Agent - Reviews and provides feedback
4. Coordinator Agent - Orchestrates the workflow
"""

import os
import json
import requests
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
import yfinance as yf
import pandas as pd

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# =============================================================================
# CUSTOM TOOLS (Beyond Tavily)
# =============================================================================

@tool
def get_stock_data(symbol: str, period: str = "1mo") -> str:
    """
    Get stock market data for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        JSON string with stock data including current price, change, volume, etc.
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return f"No data found for symbol: {symbol}"
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
        
        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "high_52w": round(hist['High'].max(), 2),
            "low_52w": round(hist['Low'].min(), 2),
            "company_name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching stock data for {symbol}: {str(e)}"


@tool
def get_economic_indicators() -> str:
    """
    Get current economic indicators from public APIs.
    
    Returns:
        JSON string with economic data like GDP, inflation, unemployment, etc.
    """
    try:
        # This is a mock implementation - in reality you'd use APIs like FRED, World Bank, etc.
        # For demo purposes, we'll return sample data
        indicators = {
            "gdp_growth": 2.1,
            "inflation_rate": 3.2,
            "unemployment_rate": 3.7,
            "interest_rate": 5.25,
            "consumer_confidence": 102.3,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Mock Economic Data API"
        }
        
        return json.dumps(indicators, indent=2)
    except Exception as e:
        return f"Error fetching economic indicators: {str(e)}"


@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of given text.
    
    Args:
        text: Text to analyze
    
    Returns:
        JSON string with sentiment analysis results
    """
    try:
        # Simple sentiment analysis (in production, use proper NLP libraries)
        positive_words = ['good', 'great', 'excellent', 'positive', 'strong', 'growth', 'profit', 'gain', 'up', 'rise']
        negative_words = ['bad', 'poor', 'negative', 'weak', 'loss', 'decline', 'down', 'fall', 'crisis', 'problem']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        result = {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "text_length": len(text),
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"


@tool
def calculate_metrics(data: str) -> str:
    """
    Calculate various financial and statistical metrics from provided data.
    
    Args:
        data: JSON string containing numerical data
    
    Returns:
        JSON string with calculated metrics
    """
    try:
        # Parse the input data
        parsed_data = json.loads(data) if isinstance(data, str) else data
        
        # Extract numerical values
        numbers = []
        for key, value in parsed_data.items():
            if isinstance(value, (int, float)) and key not in ['last_updated', 'analyzed_at']:
                numbers.append(value)
        
        if not numbers:
            return "No numerical data found to calculate metrics"
        
        # Calculate basic statistics
        metrics = {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers),
            "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Calculate standard deviation if we have enough data points
        if len(numbers) > 1:
            mean = metrics["average"]
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            metrics["std_deviation"] = variance ** 0.5
        
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Error calculating metrics: {str(e)}"


# =============================================================================
# AGENT STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_data: str
    analysis_result: str
    critique_feedback: str
    final_decision: str
    current_agent: str
    task_completed: bool


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

def create_research_agent():
    """Create a research agent with multiple tools."""
    # Set up tools
    search_tool = TavilySearchResults(max_results=3)
    tools = [search_tool, get_stock_data, get_economic_indicators]
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # Create system prompt
    system_prompt = """You are a Research Agent specialized in gathering comprehensive information.

Your responsibilities:
1. Use multiple tools to gather relevant data
2. Search the web for current information
3. Get financial/economic data when relevant
4. Compile findings into a structured format
5. Ensure data is current and from reliable sources

When researching:
- Use web search for current events and general information
- Use stock data tool for financial information
- Use economic indicators for market context
- Always verify information from multiple sources
- Present findings clearly and objectively

Format your research findings with clear sections and sources."""

    return llm.bind_tools(tools), system_prompt


def create_analysis_agent():
    """Create an analysis agent with analytical tools."""
    tools = [analyze_sentiment, calculate_metrics]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    system_prompt = """You are an Analysis Agent specialized in processing and analyzing data.

Your responsibilities:
1. Analyze research data provided by the Research Agent
2. Use sentiment analysis on textual data
3. Calculate relevant metrics and statistics
4. Identify patterns, trends, and insights
5. Provide data-driven conclusions

When analyzing:
- Look for patterns and correlations in the data
- Use sentiment analysis for qualitative data
- Calculate metrics for quantitative data
- Consider multiple perspectives
- Provide evidence-based insights
- Highlight key findings and implications

Format your analysis with clear conclusions and supporting evidence."""

    return llm.bind_tools(tools), system_prompt


def create_critic_agent():
    """Create a critic agent for reviewing and providing feedback."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    system_prompt = """You are a Critic Agent specialized in reviewing and critiquing analysis.

Your responsibilities:
1. Review the research and analysis provided
2. Identify potential biases, gaps, or weaknesses
3. Suggest improvements or additional considerations
4. Validate conclusions against evidence
5. Provide constructive feedback

When critiquing:
- Check for logical consistency
- Identify missing information or perspectives
- Question assumptions and methodology
- Suggest alternative interpretations
- Evaluate the strength of conclusions
- Provide specific, actionable feedback

Be thorough but constructive in your critique."""

    return llm, system_prompt


def create_coordinator_agent():
    """Create a coordinator agent for orchestrating the workflow."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    system_prompt = """You are a Coordinator Agent responsible for orchestrating the multi-agent workflow.

Your responsibilities:
1. Understand the user's request and break it down
2. Coordinate between different agents
3. Make decisions about workflow progression
4. Synthesize findings from all agents
5. Provide final recommendations

When coordinating:
- Clearly define the task for each agent
- Ensure all necessary information is gathered
- Integrate insights from research, analysis, and critique
- Make informed decisions based on all available data
- Provide clear, actionable final recommendations

You have the authority to request additional research or analysis if needed."""

    return llm, system_prompt


# =============================================================================
# AGENT NODES
# =============================================================================

def research_node(state: AgentState):
    """Research agent node."""
    llm, system_prompt = create_research_agent()
    
    # Get the user's request
    user_message = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    if not user_message:
        return {"current_agent": "research", "research_data": "No user request found"}
    
    # Create research prompt
    research_prompt = f"""
    {system_prompt}
    
    User Request: {user_message}
    
    Please conduct comprehensive research on this topic. Use all available tools to gather relevant information.
    """
    
    messages = [SystemMessage(content=research_prompt)]
    response = llm.invoke(messages)
    
    # If the response contains tool calls, we need to execute them
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # This is a simplified version - in a full implementation, you'd use ToolNode
        research_data = f"Research initiated for: {user_message}\nTools to be called: {len(response.tool_calls)}"
    else:
        research_data = response.content
    
    return {
        "current_agent": "research",
        "research_data": research_data,
        "messages": [AIMessage(content=f"Research Agent: {research_data}")]
    }


def analysis_node(state: AgentState):
    """Analysis agent node."""
    llm, system_prompt = create_analysis_agent()
    
    research_data = state.get("research_data", "No research data available")
    
    analysis_prompt = f"""
    {system_prompt}
    
    Research Data to Analyze:
    {research_data}
    
    Please analyze this research data thoroughly. Use sentiment analysis and calculate relevant metrics where appropriate.
    """
    
    messages = [SystemMessage(content=analysis_prompt)]
    response = llm.invoke(messages)
    
    analysis_result = response.content
    
    return {
        "current_agent": "analysis",
        "analysis_result": analysis_result,
        "messages": [AIMessage(content=f"Analysis Agent: {analysis_result}")]
    }


def critique_node(state: AgentState):
    """Critic agent node."""
    llm, system_prompt = create_critic_agent()
    
    research_data = state.get("research_data", "No research data")
    analysis_result = state.get("analysis_result", "No analysis result")
    
    critique_prompt = f"""
    {system_prompt}
    
    Research Data:
    {research_data}
    
    Analysis Result:
    {analysis_result}
    
    Please provide a thorough critique of the research and analysis. Identify strengths, weaknesses, and areas for improvement.
    """
    
    messages = [SystemMessage(content=critique_prompt)]
    response = llm.invoke(messages)
    
    critique_feedback = response.content
    
    return {
        "current_agent": "critique",
        "critique_feedback": critique_feedback,
        "messages": [AIMessage(content=f"Critic Agent: {critique_feedback}")]
    }


def coordinator_node(state: AgentState):
    """Coordinator agent node."""
    llm, system_prompt = create_coordinator_agent()
    
    research_data = state.get("research_data", "No research data")
    analysis_result = state.get("analysis_result", "No analysis result")
    critique_feedback = state.get("critique_feedback", "No critique feedback")
    
    coordinator_prompt = f"""
    {system_prompt}
    
    Research Data:
    {research_data}
    
    Analysis Result:
    {analysis_result}
    
    Critique Feedback:
    {critique_feedback}
    
    Please synthesize all the information and provide final recommendations and decisions.
    """
    
    messages = [SystemMessage(content=coordinator_prompt)]
    response = llm.invoke(messages)
    
    final_decision = response.content
    
    return {
        "current_agent": "coordinator",
        "final_decision": final_decision,
        "task_completed": True,
        "messages": [AIMessage(content=f"Coordinator Agent: {final_decision}")]
    }


# =============================================================================
# WORKFLOW ROUTING
# =============================================================================

def route_workflow(state: AgentState) -> Literal["research", "analysis", "critique", "coordinator", "__end__"]:
    """Route the workflow based on current state."""
    current_agent = state.get("current_agent", "")
    
    if current_agent == "":
        return "research"
    elif current_agent == "research":
        return "analysis"
    elif current_agent == "analysis":
        return "critique"
    elif current_agent == "critique":
        return "coordinator"
    elif current_agent == "coordinator":
        return "__end__"
    else:
        return "__end__"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_multi_agent_graph():
    """Create the multi-agent workflow graph."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("coordinator", coordinator_node)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add edges
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "critique")
    workflow.add_edge("critique", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable is not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🚀 Advanced Multi-Agent System Starting...")
    print("=" * 60)
    
    # Create the workflow
    app = create_multi_agent_graph()
    
    # Example queries to test the system
    test_queries = [
        "Analyze the current state of the electric vehicle market and Tesla's position",
        "Research the impact of recent Federal Reserve decisions on the stock market",
        "Investigate the potential of renewable energy investments in 2024"
    ]
    
    print("Available test queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    
    print("\nOr enter your own query:")
    
    # Get user input
    try:
        choice = input("\nEnter query number (1-3) or type your own query: ").strip()
        
        if choice in ['1', '2', '3']:
            user_query = test_queries[int(choice) - 1]
        else:
            user_query = choice if choice else test_queries[0]
        
        print(f"\n🔍 Processing query: {user_query}")
        print("=" * 60)
        
        # Run the workflow
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "research_data": "",
            "analysis_result": "",
            "critique_feedback": "",
            "final_decision": "",
            "current_agent": "",
            "task_completed": False
        }
        
        # Stream the results
        for step, output in enumerate(app.stream(initial_state), 1):
            print(f"\n📍 Step {step}: {output.get('current_agent', 'Unknown').title()} Agent")
            print("-" * 40)
            
            if 'messages' in output and output['messages']:
                latest_message = output['messages'][-1]
                if hasattr(latest_message, 'content'):
                    print(latest_message.content[:500] + "..." if len(latest_message.content) > 500 else latest_message.content)
            
            print()
        
        print("✅ Multi-Agent Analysis Complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
