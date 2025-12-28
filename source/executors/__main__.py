from source.executors import SimpleExecutor

if __name__ == "__main__":
    executor = SimpleExecutor(
        model="openai:gpt-4o-mini",
        system_prompt="You are a simple executor that can execute a query",
        tools=[],
    )
    print(executor.execute("What is the weather in Tokyo?"))

