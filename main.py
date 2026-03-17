from agent.query_agent import agent_executor

def main():
    """
    This is the main entry point for the Dubai Free Zone Research Assistant.
    It captures user input from the terminal and passes it to Agent 1.
    """
    print("===============================================")
    print("   SHARJAH FREE ZONE RESEARCH ASSISTANT (v1.0)   ")
    print("===============================================")
    print("Instructions:")
    print("- Ask about company counts (e.g., 'How many active companies in JAFZA?')")
    print("- Type 'exit' or 'quit' to close the program.")
    print("-----------------------------------------------")

    while True:
        # Capture user input
        user_query = input("\n[User]: ")

        # Check if the user wants to quit
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Exiting Assistant. Goodbye!")
            break
        
        # Handle empty input
        if not user_query.strip():
            continue

        print("\n[Agent]: Thinking...")

        try:
            # Invoke the agent
            # This sends 'user_query' into the {input} placeholder in query_agent.py
            response = agent_executor.invoke({"input": user_query})
            
            # Display the agents final response
            print(f"\n[Agent]: {response['output']}")

        except Exception as e:
            # Catching errors like missing API keys or connection issues
            print(f"\n[Error]: Something went wrong. Details: {e}")
            print("Ensure your mock_api.py is running and your API key is correct.")

if __name__ == "__main__":
    main()