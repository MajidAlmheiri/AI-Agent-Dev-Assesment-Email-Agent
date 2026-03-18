import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools.api_tools import count_companies, get_setup_estimates

# We keep configuration in .env so it is easy to switch keys/models
# without touching code every time.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

# Read API key from environment (loaded from .env above).
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "Missing GOOGLE_API_KEY. Add it to your .env file at the project root or export it in your shell."
    )

# Pick a model from env so we can test different Gemini variants quickly.
model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

# Bind both of our tools so the model can call them when needed.
llm_with_tools = llm.bind_tools([count_companies, get_setup_estimates])


def _content_to_text(content) -> str:
    # Gemini can return content in structured blocks (list of dicts).
    # This helper normalizes everything into plain text for clean CLI output.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        if text_parts:
            return "\n".join(text_parts).strip()
        return str(content)
    return str(content)

class ResilientAgentExecutor:
    """
    Small adapter around the model so main.py can call:
    agent_executor.invoke({"input": "..."})

    It runs a lightweight 2-pass loop:
    1) model decides whether to call tool
    2) if tool is called, we execute it and ask model for final wording
    """
    def __init__(self):
        # I noticed the agent forgets what we just talked about in previous emails.
        # So I am saving the chat history in this memory list.
        # I hope I understood how to keep the memory running correctly!
        self.memory = [
            SystemMessage(content=(
                "You are a professional Business Consultant for Sharjah Free Zones. "
                "Your primary task is to respond to email inquiries from potential investors. "
                "\nINSTRUCTIONS:"
                "\n1. Use 'get_setup_estimates' for questions about costs, capital, fees, or setup requirements."
                "\n2. Use 'count_companies' if the user asks for market statistics."
                "\n3. CRITICAL: When passing the 'industry' argument to tools, extract ONLY a single, broad keyword. For example, if the user asks for 'general trading company', pass 'trading'. If they ask for 'plastic bags shop', pass 'plastic'."
                "\n4. ALWAYS respond in a structured professional email format."
                "\n5. The user might use abbreviations (e.g., JAFZA). CRITICAL: You must resolve these to their full official names (e.g., 'Jebel Ali Free Zone') BEFORE passing them to any tool."
                "\n6. If the user query is completely vague, ask for clarification politely." 
            ))
        ]

    def invoke(self, inputs: dict) -> dict:
        # Grab user text from main.py payload.
        user_query = inputs.get("input", "")
        
        # Adding the user's new email to our memory list so the ai doesn't forget it
        self.memory.append(HumanMessage(content=user_query))
        
        # First model pass: either answer directly or request tool call.
        response = llm_with_tools.invoke(self.memory)
        
        # If the model requested tools, execute each call and feed results back.
        if response.tool_calls:
            self.memory.append(response)
            
            for tool_call in response.tool_calls:
                
                # I added a check here to trigger the right tool since we have two now
                tool_name = tool_call["name"]
                if tool_name == "count_companies":
                    result = count_companies.invoke(tool_call["args"])
                elif tool_name == "get_setup_estimates":
                    result = get_setup_estimates.invoke(tool_call["args"])
                else:
                    result = f"Error: Tool {tool_name} not found."
                
                # Attach tool output to conversation so model can finalize response.
                self.memory.append(ToolMessage(
                    content=str(result), 
                    tool_call_id=tool_call["id"]
                ))
            
            # Second model pass: convert raw tool output into user-friendly text.
            response = llm_with_tools.invoke(self.memory)
            
        # Adding the final ai email to memory so it remembers its own replies for the next question
        self.memory.append(response)

        # Always return plain text (not raw structured content blocks).
        return {"output": _content_to_text(response.content)}

# Export the object that main.py imports and invokes.
agent_executor = ResilientAgentExecutor()

print("Agent 2 (Email Inquiry w/ Memory) is ready...")