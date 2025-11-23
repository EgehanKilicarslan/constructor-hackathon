import logging
import re
from pathlib import Path

import fitz
import requests

logger = logging.getLogger(__name__)


class ArticleAnalyser:
    def __init__(self, url: str, filename: str = "article.pdf", save_path: Path = Path("/tmp")):
        self.url = url
        self.filename = filename
        self.save_path = save_path

    def __enter__(self):
        logger.debug(f"Downloading article from {self.url}")
        response = requests.get(self.url)
        response.raise_for_status()

        # ensure save directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)

        file_path = self.save_path / self.filename
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.debug(f"Article downloaded and saved to {file_path}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        file_path = self.save_path / self.filename
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up downloaded article at {file_path}")

    def analyze_github_links(self) -> set[str] | None:
        github_links = set()
        regex_pattern = r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"

        try:
            file_path = self.save_path / self.filename
            with fitz.open(file_path) as doc:
                for page in doc:
                    links = page.get_links()
                    if links:
                        for link in links:
                            uri = link.get("uri", "")
                            match = re.search(regex_pattern, uri)
                            if match:
                                github_links.add(match.group(0))
            logger.debug(f"Extracted GitHub links: {github_links}")
        except Exception as e:
            logger.error(f"Error analyzing article at {file_path}: {e}")
        return github_links
