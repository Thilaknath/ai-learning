import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# --- Set up the LLM ---
# Make sure your OPENAI_API_KEY is set as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Set up the Tools ---
# We also need Tavily for the search tool
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# --- Create the Agent ---
# Using the modern create_react_agent function
app = create_react_agent(llm, tools)


# --- 6. Run the Agent! ---
if __name__ == '__main__':
    # You'll need to set your API keys as environment variables
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    # os.environ["TAVILY_API_KEY"] = "tvly-..."
    
    # Let's ask a question that requires a search
    inputs = {"messages": [HumanMessage(content="What is the current price of gold per ounce in USD?")]}
    
    # Invoke the app and stream the output to see the steps
    for output in app.stream(inputs):
        # The 'output' contains the state of the graph at each step
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
