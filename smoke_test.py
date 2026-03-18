import os

os.environ.setdefault("LLM_PROVIDER", "rule")

from agent.query_agent import agent_executor


def run_smoke_test() -> None:
    query_1 = "How many active logistics companies are registered in JAFZA?"
    res_1 = agent_executor.invoke({"input": query_1})
    print("\n[Smoke Test] Agent 1-style query")
    print(res_1["output"])

    query_2 = (
        "I am looking to open a logistics company in Jebel Ali Free Zone. "
        "I will hire exactly 15 employees. Can you provide estimated setup costs?"
    )
    res_2 = agent_executor.invoke({"input": query_2})
    print("\n[Smoke Test] Agent 2-style inquiry")
    print(res_2["output"])


if __name__ == "__main__":
    run_smoke_test()
