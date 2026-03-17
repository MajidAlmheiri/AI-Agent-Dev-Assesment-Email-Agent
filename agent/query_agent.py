import os
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.api_tools import count_companies
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# You can change how the AI model behaves by changing the system_prompt
# The human_prompt is what the user is going to ask the agent
system_prompt = "You are a helpful assistant that provides detailed explanations."
human_prompt = input("")

# API Key setup 
os.environ["GOOGLE_API_KEY"] = "AIzaSyAS9b73JPLGNGCpAEf11r9ofb1qB3Rd_50"

# We use the Flash model because it's fast and free
agent = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# These are the tools (methods) that our AI agent will be able to use
tools = [count_companies]

print("Agent setup started...")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])