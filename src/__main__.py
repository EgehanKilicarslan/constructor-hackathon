from pathlib import Path

from service import ArticleAnalyser, ChatModel, GithubAnalyser

if __name__ == "__main__":
    model = ChatModel()

    with ArticleAnalyser(url="https://arxiv.org/pdf/2507.06825") as analyser:
        model.add_document(analyser.file_path)
        gh_links = analyser.analyze_github_links()
        if gh_links is None:
            raise ValueError("No GitHub links found in the article.")
        for gh_link in gh_links:
            print("Analyzing GitHub link:", gh_link)
            with GithubAnalyser(repo_url=gh_link) as gh_analyser:
                files: dict[str, list[Path]] = gh_analyser.identify_and_download_key_files()
                for category, paths in files.items():
                    for path in paths:
                        model.add_document(Path(path))

    response = model.send(
        "Can you give me a demo code according to the files given and the article and make it in python and add docstrings?"
    )
    print("Response:", response)
