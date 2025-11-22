from adapter import StatelessConstructorAdapter

if __name__ == "__main__":
    adapterStateless = StatelessConstructorAdapter()

    print("Retrieving available LLMs Stateful")
    available_llms = adapterStateless.get_available_llms()
    for llm in available_llms.results:
        print(f"LLM Alias: {llm.alias}, Name: {llm.name}, ID: {llm.id}")
