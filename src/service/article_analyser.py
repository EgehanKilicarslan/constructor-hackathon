import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
import requests

logger = logging.getLogger(__name__)


class ArticleAnalyser:
    def __init__(self, url: str, filename: str = "article.pdf", save_path: Path = Path("/tmp")):
        self.url = url
        self.filename = filename
        self.save_path = save_path
        self.file_path = self.save_path / self.filename

    def __enter__(self):
        """Context manager entry: Downloads the file."""
        self.download()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit: Cleans up the file."""
        self.cleanup()

    def download(self) -> None:
        """Downloads the file. (Public method, callable externally)"""
        if self.file_path.exists():
            logger.info(f"File {self.file_path} already exists. Skipping download.")
            return

        logger.debug(f"Downloading article from {self.url}")
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()

            self.save_path.mkdir(parents=True, exist_ok=True)

            with open(self.file_path, "wb") as f:
                f.write(response.content)
            logger.debug(f"Article downloaded and saved to {self.file_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to download article from {self.url}: {e}")
            raise RuntimeError(f"Failed to download article: {e}") from e

    def cleanup(self):
        """Deletes the downloaded file. (Public method)"""
        try:
            if self.file_path.exists():
                self.file_path.unlink()
                logger.debug(f"Cleaned up downloaded article at {self.file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {self.file_path}: {e}")

    def analyze_github_links(self) -> set[str] | None:
        """Finds GitHub links inside the PDF."""
        github_links = set()
        regex_pattern = r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"

        if not self.file_path.exists():
            logger.info("File not found via manual check, downloading now...")
            self.download()

        try:
            with fitz.open(self.file_path) as doc:
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
            logger.error(f"Error analyzing article at {self.file_path}: {e}")
            return None  # You can return None or an empty set in case of error

        return github_links
