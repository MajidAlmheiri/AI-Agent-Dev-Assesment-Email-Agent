import os
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.api_tools import count_companies
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# You can change how the AI model behaves by changing the system_prompt
system_prompt = "You are a helpful assistant that provides detailed explanations."


# API Key setup 
os.environ["GOOGLE_API_KEY"] = "AIzaSyAS9b73JPLGNGCpAEf11r9ofb1qB3Rd_50"

# We use the Flash model because it's fast and free
query_agent = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# These are the tools (methods) that our AI agent will be able to use
tools = [count_companies]

print("Agent setup started...")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# Assembling the agent
agent_logic = create_tool_calling_agent(query_agent, tools, chat_prompt)

# 6. CREATE THE EXECUTOR (The manager)
# This is what main.py imports!
agent_executor = AgentExecutor(agent=agent_logic, tools=tools, verbose=True)

print("Agent is ready...")