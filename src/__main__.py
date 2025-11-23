from pathlib import Path

from service import ArticleAnalyser, GithubAnalyser

if __name__ == "__main__":
    with ArticleAnalyser(
        "https://arxiv.org/pdf/2203.14090",
        save_path=Path("/home/egehan/Workspace/constructor-hackathon/tmp"),
    ) as analyser:
        github_links = analyser.analyze_github_links()
        for link in github_links or []:
            with GithubAnalyser(
                link, save_path=Path("/home/egehan/Workspace/constructor-hackathon/tmp")
            ) as gh_analyser:
                repo_structure = gh_analyser.get_repo_structure()
                downloaded_files = gh_analyser.identify_and_download_key_files()
                print(f"Repository Structure for {link}: {repo_structure}")
                print(f"Downloaded Key Files for {link}: {downloaded_files}")
