import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from tools.api_tools import count_companies, get_setup_estimates


BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=False)


def _content_to_text(content: Any) -> str:
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


def _extract_headcount(query: str) -> int:
    match = re.search(r"\b(\d{1,4})\s*(employees?|staff|people|visas?)\b", query, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def _extract_headcounts(query: str) -> list[int]:
    matches = re.findall(r"\b(\d{1,4})\s*(?:employees?|staff|people|visas?)\b", query, flags=re.IGNORECASE)
    headcounts: list[int] = []
    seen: set[int] = set()
    for value in matches:
        count = int(value)
        if count not in seen:
            seen.add(count)
            headcounts.append(count)
    return headcounts


def _normalize_freezone(query: str) -> str:
    normalized = query.lower()

    aliases = {
        "jafza": "Jebel Ali Free Zone",
        "jebel ali free zone": "Jebel Ali Free Zone",
        "saif zone": "Sharjah Airport International Free Zone",
        "saif": "Sharjah Airport International Free Zone",
        "hamriyah free zone": "Hamriyah Free Zone",
        "hfza": "Hamriyah Free Zone",
    }

    for key, value in aliases.items():
        if key in normalized:
            return value

    fallback = re.search(r"in\s+([a-zA-Z\s]+free\s+zone)", query, flags=re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip()

    return "Jebel Ali Free Zone"


def _extract_freezone_candidate(query: str) -> str | None:
    normalized = query.lower()
    aliases = {
        "jafza": "Jebel Ali Free Zone",
        "jebel ali free zone": "Jebel Ali Free Zone",
        "saif zone": "Sharjah Airport International Free Zone",
        "saif": "Sharjah Airport International Free Zone",
        "hamriyah free zone": "Hamriyah Free Zone",
        "hfza": "Hamriyah Free Zone",
    }

    for key, value in aliases.items():
        if key in normalized:
            return value

    fallback = re.search(r"in\s+([a-zA-Z\s]+free\s+zone)", query, flags=re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip()

    return None


def _extract_industry(query: str) -> str:
    q = query.lower()

    known_keywords = [
        "logistics",
        "trading",
        "cargo",
        "freight",
        "manufacturing",
        "retail",
        "food",
        "technology",
        "it",
        "fintech",
        "energy",
        "consulting",
        "chemical",
    ]

    for keyword in known_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", q):
            return keyword

    pattern_1 = re.search(r"open\s+(?:a|an)\s+([a-zA-Z\-\s]+?)\s+company", q)
    if pattern_1:
        phrase = pattern_1.group(1).strip().split()
        if phrase:
            return phrase[-1]

    pattern_2 = re.search(r"for\s+([a-zA-Z\-\s]+?)\s+(?:company|business)", q)
    if pattern_2:
        phrase = pattern_2.group(1).strip().split()
        if phrase:
            return phrase[-1]

    return "trading"


def _extract_industry_candidate(query: str) -> str | None:
    q = query.lower()

    known_keywords = [
        "logistics",
        "trading",
        "cargo",
        "freight",
        "manufacturing",
        "retail",
        "food",
        "technology",
        "it",
        "fintech",
        "energy",
        "consulting",
        "chemical",
    ]

    for keyword in known_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", q):
            return keyword

    pattern_1 = re.search(r"open\s+(?:a|an)\s+([a-zA-Z\-\s]+?)\s+company", q)
    if pattern_1:
        phrase = pattern_1.group(1).strip().split()
        if phrase:
            return phrase[-1]

    pattern_2 = re.search(r"for\s+([a-zA-Z\-\s]+?)\s+(?:company|business)", q)
    if pattern_2:
        phrase = pattern_2.group(1).strip().split()
        if phrase:
            return phrase[-1]

    return None


def _is_cost_query(query: str) -> bool:
    return bool(
        re.search(
            r"cost|estimate|setup|capital|budget|visa|open|employees?|staff|people|what about",
            query,
            flags=re.IGNORECASE,
        )
    )


def _is_count_query(query: str) -> bool:
    return bool(re.search(r"how many|count|number of", query, flags=re.IGNORECASE))


def _format_email_reply(estimate: dict, freezone: str, industry: str) -> str:
    if "error" in estimate:
        return (
            "Subject: Free Zone Setup Cost Estimate\n\n"
            "Dear Investor,\n\n"
            f"Thank you for your inquiry. I could not find enough matching records for '{industry}' in '{freezone}'. "
            "Please share a broader activity keyword (for example: trading, logistics, food, technology) so I can provide a precise estimate.\n\n"
            "Kind regards,\n"
            "Free Zone Advisory Team"
        )

    return (
        "Subject: Estimated Business Setup Cost in Free Zone\n\n"
        "Dear Investor,\n\n"
        "Thank you for your inquiry. Based on the available free zone company dataset, please find your estimate below:\n\n"
        f"- Free Zone: {estimate.get('matched_freezone', freezone)}\n"
        f"- Industry Keyword: {estimate.get('matched_industry', industry)}\n"
        f"- Average License Fee: {estimate.get('average_license_fee_estimate', estimate.get('license_fee_estimate', 'N/A'))}\n"
        f"- Average Capital Requirement: {estimate.get('average_capital', 'N/A')}\n"
        f"- Estimated Visa Budget: {estimate.get('visa_budget_estimate', 'N/A')}\n"
        f"- Estimated Total Setup Cost: {estimate.get('total_estimated_setup', 'N/A')}\n"
        f"- Estimated Setup Range: {estimate.get('setup_cost_range', 'N/A')}\n"
        f"- Staffing Basis: {estimate.get('employee_note', 'N/A')}\n\n"
        "Please note these are indicative planning estimates and may vary based on license type, office package, and authority updates.\n\n"
        "Kind regards,\n"
        "Free Zone Advisory Team"
    )


def _append_comparison_section(base_email: str, comparison_estimate: dict, comparison_headcount: int) -> str:
    if "error" in comparison_estimate:
        return base_email

    comparison_block = (
        "\n\nAdditional Scenario (Comparison)\n"
        f"- Headcount: {comparison_headcount} employees\n"
        f"- Estimated Visa Budget: {comparison_estimate.get('visa_budget_estimate', 'N/A')}\n"
        f"- Estimated Total Setup Cost: {comparison_estimate.get('total_estimated_setup', 'N/A')}\n"
        f"- Estimated Setup Range: {comparison_estimate.get('setup_cost_range', 'N/A')}\n"
    )

    signature_marker = "\n\nKind regards,\nFree Zone Advisory Team"
    if signature_marker in base_email:
        return base_email.replace(signature_marker, comparison_block + signature_marker)
    return base_email + comparison_block


def _rule_based_response(
    user_query: str,
    default_freezone: str | None = None,
    default_industry: str | None = None,
) -> tuple[str, str, str]:
    freezone = _extract_freezone_candidate(user_query) or default_freezone or _normalize_freezone(user_query)
    industry = _extract_industry_candidate(user_query) or default_industry or _extract_industry(user_query)
    headcounts = _extract_headcounts(user_query)
    headcount = headcounts[0] if headcounts else _extract_headcount(user_query)

    if _is_count_query(user_query):
        status = "Active"
        count = count_companies.invoke({"freezone": freezone, "status": status, "industry": industry})
        return f"There are {count} {status.lower()} companies matching '{industry}' in {freezone}.", freezone, industry

    if _is_cost_query(user_query):
        primary_estimate = get_setup_estimates.invoke(
            {"freezone": freezone, "industry": industry, "target_employees": headcount}
        )
        email_output = _format_email_reply(primary_estimate, freezone, industry)

        if len(headcounts) >= 2:
            comparison_headcount = headcounts[1]
            comparison_estimate = get_setup_estimates.invoke(
                {
                    "freezone": freezone,
                    "industry": industry,
                    "target_employees": comparison_headcount,
                }
            )
            email_output = _append_comparison_section(email_output, comparison_estimate, comparison_headcount)

        return email_output, freezone, industry

    return (
        "Subject: Clarification Required for Free Zone Inquiry\n\n"
        "Dear Investor,\n\n"
        "Thank you for your message. Could you please confirm the free zone, industry keyword, and expected headcount so I can provide an accurate estimate?\n\n"
        "Kind regards,\n"
        "Free Zone Advisory Team"
    ), freezone, industry


def _normalize_tool_args(raw_args: dict) -> dict:
    if not isinstance(raw_args, dict):
        return {}

    if "parameters" in raw_args and isinstance(raw_args["parameters"], dict):
        return raw_args["parameters"]

    if "arguments" in raw_args and isinstance(raw_args["arguments"], dict):
        return raw_args["arguments"]

    return raw_args


def _looks_like_count_answer(text: str) -> bool:
    if not text:
        return False
    has_number = bool(re.search(r"\b\d+\b", text))
    mentions_companies = "companies" in text.lower() or "company" in text.lower()
    return has_number and mentions_companies


def _looks_like_email_estimate(text: str) -> bool:
    if not text:
        return False
    lower_text = text.lower()
    required_markers = ["subject:", "dear", "estimated", "kind regards"]
    return all(marker in lower_text for marker in required_markers)


def _looks_like_tool_json(text: str) -> bool:
    lower_text = text.lower()
    return (
        '"name": "get_setup_estimates"' in lower_text
        or '"name": "count_companies"' in lower_text
        or ('"parameters"' in lower_text and '{' in lower_text and '}' in lower_text)
    )


def _build_llm_with_tools():
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()

    if provider == "rule":
        return None, provider

    if provider == "ollama":
        try:
            import importlib

            ChatOllama = importlib.import_module("langchain_ollama").ChatOllama
        except Exception as exc:
            raise RuntimeError(
                "LLM_PROVIDER=ollama but langchain-ollama is not installed. Run: pip install langchain-ollama"
            ) from exc

        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        llm = ChatOllama(model=ollama_model, temperature=0)
        return llm.bind_tools([count_companies, get_setup_estimates]), provider

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (or GEMINI_API_KEY). Set LLM_PROVIDER=rule for no-LLM mode."
        )
    if api_key.strip() == "your_new_google_ai_studio_key_here":
        raise RuntimeError("API key placeholder detected. Replace with a real key or use LLM_PROVIDER=rule.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
    return llm.bind_tools([count_companies, get_setup_estimates]), provider


class ResilientAgentExecutor:
    def __init__(self):
        self.memory = [
            SystemMessage(
                content=(
                    "You are a professional Business Consultant for UAE Free Zones. "
                    "Use tools for factual answers. "
                    "For count/statistics questions, call count_companies and reply naturally. "
                    "For setup/cost questions, call get_setup_estimates and respond in a structured professional email format. "
                    "Extract a single broad industry keyword when needed. "
                    "Map abbreviations like JAFZA to Jebel Ali Free Zone."
                )
            )
        ]
        self.llm_with_tools, self.provider = _build_llm_with_tools()
        self.last_freezone = "Jebel Ali Free Zone"
        self.last_industry = "trading"

    def _invoke_model_path(self, user_query: str) -> str:
        self.memory.append(HumanMessage(content=user_query))
        response = self.llm_with_tools.invoke(self.memory)

        if response.tool_calls:
            self.memory.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = _normalize_tool_args(tool_call.get("args", {}))
                if tool_name == "count_companies":
                    result = count_companies.invoke(tool_args)
                elif tool_name == "get_setup_estimates":
                    result = get_setup_estimates.invoke(tool_args)
                else:
                    result = f"Error: Tool {tool_name} not found."

                self.memory.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            response = self.llm_with_tools.invoke(self.memory)

        elif _is_count_query(user_query) or _is_cost_query(user_query):
            fallback_output, _, _ = _rule_based_response(
                user_query,
                default_freezone=self.last_freezone,
                default_industry=self.last_industry,
            )
            return fallback_output

        self.memory.append(response)
        return _content_to_text(response.content)

    def invoke(self, inputs: dict) -> dict:
        user_query = inputs.get("input", "")

        if self.provider == "rule" or self.llm_with_tools is None:
            output, freezone, industry = _rule_based_response(
                user_query,
                default_freezone=self.last_freezone,
                default_industry=self.last_industry,
            )
            self.last_freezone = freezone
            self.last_industry = industry
            return {"output": output}

        try:
            model_output = self._invoke_model_path(user_query)

            if _looks_like_tool_json(model_output):
                output, freezone, industry = _rule_based_response(
                    user_query,
                    default_freezone=self.last_freezone,
                    default_industry=self.last_industry,
                )
                self.last_freezone = freezone
                self.last_industry = industry
                return {"output": output}

            if _is_count_query(user_query) and not _looks_like_count_answer(model_output):
                output, freezone, industry = _rule_based_response(
                    user_query,
                    default_freezone=self.last_freezone,
                    default_industry=self.last_industry,
                )
                self.last_freezone = freezone
                self.last_industry = industry
                return {"output": output}

            if _is_cost_query(user_query) and not _looks_like_email_estimate(model_output):
                output, freezone, industry = _rule_based_response(
                    user_query,
                    default_freezone=self.last_freezone,
                    default_industry=self.last_industry,
                )
                self.last_freezone = freezone
                self.last_industry = industry
                return {"output": output}

            freezone_candidate = _extract_freezone_candidate(user_query)
            industry_candidate = _extract_industry_candidate(user_query)
            if freezone_candidate:
                self.last_freezone = freezone_candidate
            if industry_candidate:
                self.last_industry = industry_candidate

            return {"output": model_output}
        except Exception as exc:
            error_text = str(exc)
            if any(marker in error_text for marker in ["RESOURCE_EXHAUSTED", "429", "API_KEY_INVALID", "quota"]):
                fallback, freezone, industry = _rule_based_response(
                    user_query,
                    default_freezone=self.last_freezone,
                    default_industry=self.last_industry,
                )
                self.last_freezone = freezone
                self.last_industry = industry
                notice = (
                    "[Notice: LLM API unavailable due to quota/key issue. Generated a deterministic dataset-based response.]\n\n"
                )
                return {"output": notice + fallback}
            raise


agent_executor = ResilientAgentExecutor()
print(f"Agent 2 (Email Inquiry w/ Memory) is ready... provider={agent_executor.provider}")