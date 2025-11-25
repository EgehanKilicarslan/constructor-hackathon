import argparse
import os

from agent import AgentState, app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start agent with an article link")
    parser.add_argument("-link", "--link", dest="link", help="article URL", default=None)
    args, _ = parser.parse_known_args()

    url = args.link or os.environ.get("LINK") or "https://github.com/strakam/generals-bots"

    initial_state: AgentState = {
        "article_url": url,
        "github_links": [],
        "file_paths": [],
        "ai_response": "",
    }

    print(f"Starting the agent graph with URL: {url}")

    # Start the agent application with the initial state
    app.invoke(initial_state)

    print("Agent run completed.")
