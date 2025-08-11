import os
from langchain_core.messages import HumanMessage
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core import get_proxy_client
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# --- Set up the LLM ---
# Make sure your OPENAI_API_KEY is set as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
proxy_client = get_proxy_client('gen-ai-hub')
llm = ChatOpenAI(proxy_model_name="gpt-4o", proxy_client=proxy_client, temperature=0)

