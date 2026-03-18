import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from tools.api_tools import count_companies, get_setup_estimates


# ----------------------------
# Environment / configuration
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=False)


# ----------------------------
# Helper: safely extract text from an AIMessage
# ----------------------------
def _content_to_text(content) -> str:
    """Convert AIMessage.content (str or list of dicts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
        )
    return str(content)


# ----------------------------
# System prompt - the single source of agent behaviour
# ----------------------------
SYSTEM_PROMPT = SystemMessage(content="""You are a Professional Business Consultant specializing in UAE Free Zones.

## YOUR ROLE
You help investors understand company setup costs in Free Zones by querying a dataset
of real registered companies and returning structured, professional email responses.

## RESPONSE FORMAT
Always reply as a professional email. Include a descriptive Subject line, a "Dear Investor," greeting, a friendly opening, and sign off with "Kind regards, Free Zone Advisory Team". 
Present the cost breakdown clearly using bullet points. Do not use placeholder brackets like < > in your final text.

## CRITICAL ALIAS RULE
BEFORE calling any tool, you MUST translate common abbreviations to their full official names:
- JAFZA  → Jebel Ali Free Zone
- SAIF / SAIF Zone → Sharjah Airport International Free Zone
- HFZA  → Hamriyah Free Zone
Always pass the FULL name to the tool, never the abbreviation.

## CRITICAL KEYWORD RULE
When calling tools, extract ONLY a single broad industry keyword from the user's description.
Examples:
- "plastic bag maker" → use keyword "plastic"
- "import/export trading company" → use keyword "trading"
- "food packaging business" → use keyword "food"
- "IT consulting firm" → use keyword "consulting"

## TOOL USAGE
You have two tools:
1. `count_companies` — counts companies matching a freezone, status, and industry keyword.
   Use when the user asks "how many companies…" or similar counting questions.
2. `get_setup_estimates` — returns average capital, license fees, visa budgets, and cost ranges.
   Use when the user asks about costs, setup, estimates, or budget.
   If the user specifies a headcount (e.g., "15 employees"), pass it as `target_employees`.
   If no headcount is mentioned, pass `target_employees=0` to use the industry average.

## EXTREME ANTI-HALLUCINATION RULES
Read these IF/THEN rules carefully before responding:
- IF the `get_setup_estimates` tool returns actual numbers, use them to write the email.
- IF the `get_setup_estimates` tool returns an error or says "No matching data found", YOU ARE STRICTLY FORBIDDEN from guessing or providing general estimates.
- IF data is missing, you MUST reply with exactly this sentiment: "I apologize, but I could not find specific data for that industry in our database. Could you please provide a broader industry keyword?"
- NEVER invent, fabricate, or guess capital requirements, license fees, or visa costs.
- IF the user asks a follow-up, use the conversation history to maintain context.
""")


# ----------------------------
# The tools list bound to the LLM
# ----------------------------
TOOLS = [count_companies, get_setup_estimates]


class ResilientAgentExecutor:
    """
    LLM-driven agent with conversational memory and a tool-calling loop.
    Uses ChatOllama to parse intent, invoke tools, and compose
    professional email responses — no hardcoded regex or formatting logic.
    """

    def __init__(self):
        # Initialise the local Ollama LLM with tool bindings
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=0.2,
        ).bind_tools(TOOLS)

        # Conversational memory: a running list of messages
        self.memory: list = [SYSTEM_PROMPT]

        # Map tool names → callable tool objects for execution
        self._tool_map = {t.name: t for t in TOOLS}

    def invoke(self, inputs: dict) -> dict:
        """
        Standard LangChain-style tool-calling loop:
        1. Append user message to memory.
        2. Call the LLM.
        3. If the LLM requests tool calls, execute them, append ToolMessages,
           and call the LLM again so it can compose the final answer.
        4. Append the final AI response to memory and return it.
        """
        query = inputs.get("input", "")

        # Step 1 — add the user's message to conversation history
        self.memory.append(HumanMessage(content=query))

        # Step 2 — call the LLM (may return tool_calls or a direct answer)
        ai_response = self.llm.invoke(self.memory)
        self.memory.append(ai_response)

        # Step 3 — tool-calling loop: keep going while the LLM asks for tools
        while ai_response.tool_calls:
            # Execute every tool the LLM requested
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_fn = self._tool_map.get(tool_name)

                if tool_fn:
                    # Run the tool and capture the result
                    result = tool_fn.invoke(tool_args)
                    self.memory.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
                else:
                    # Unknown tool — let the LLM know
                    self.memory.append(
                        ToolMessage(
                            content=f"Error: tool '{tool_name}' not found.",
                            tool_call_id=tool_call["id"],
                        )
                    )

            # Call the LLM again with the tool results so it can compose the answer
            ai_response = self.llm.invoke(self.memory)
            self.memory.append(ai_response)

        # Step 4 — extract the final text and return
        output = _content_to_text(ai_response.content)
        return {"output": output}


# ----------------------------
# Module-level instance (imported by main.py)
# ----------------------------
agent_executor = ResilientAgentExecutor()
print("Agent 2 (Email Inquiry w/ Memory) is ready  —  LLM-driven mode (Ollama local)")