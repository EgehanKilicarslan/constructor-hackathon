import base64
import logging
import re
import shutil
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class GithubAnalyser:
    def __init__(self, repo_url: str, token: str | None = None, save_path: Path = Path("/tmp")):
        """
        Initializes the GithubAnalyser.

        Args:
            repo_url: Full URL to the GitHub repository.
            token: GitHub Personal Access Token (PAT) for higher API rate limits.
            save_path: Base directory where the repo folder will be created.
        """
        self.repo_url = repo_url.rstrip("/")
        self.token = token
        self.owner, self.repo_name = self._parse_repo_url()

        # Create a unique path for this repo to avoid conflicts/overwrites
        self.save_dir = save_path / f"{self.owner}_{self.repo_name}"

        # Session will be initialized in __enter__
        self.session: requests.Session | None = None
        self.api_base = f"https://api.github.com/repos/{self.owner}/{self.repo_name}"

    def __enter__(self):
        """
        Context Manager Entry:
        1. Starts a persistent HTTP session for performance.
        2. Creates the directory structure on disk.
        """
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github.v3+json"})

        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})

        # Ensure the directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Started analysis session for {self.repo_name}. Save path: {self.save_dir}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context Manager Exit:
        1. Closes the HTTP session.
        2. Cleans up (deletes) the downloaded files and folder.
        """
        if self.session:
            self.session.close()

        # Cleanup: Remove the entire directory tree
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
            logger.info(f"Cleanup successful: Removed {self.save_dir}")

        if exc_type:
            logger.error(f"An error occurred during Github analysis: {exc_value}")

    def _parse_repo_url(self) -> tuple[str, str]:
        """Parses the owner and repository name from the URL."""
        pattern = r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
        match = re.search(pattern, self.repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {self.repo_url}")
        return match.group(1), match.group(2)

    def _ensure_session(self) -> requests.Session:
        """Helper to ensure session is active for type safety and to prevent misuse."""
        if self.session is None:
            raise RuntimeError(
                "Session is not initialized. Please use the 'with GithubAnalyser(...) as ...' context manager pattern."
            )
        return self.session

    def _get_default_branch(self) -> str:
        """
        Determines the default branch (main, master, etc.) via API.
        If API fails, it attempts to guess common names.
        """
        session = self._ensure_session()

        try:
            # The repo metadata endpoint contains the 'default_branch' key
            response = session.get(self.api_base)
            response.raise_for_status()
            branch = response.json().get("default_branch", "main")
            logger.debug(f"Detected default branch: {branch}")
            return branch
        except Exception as e:
            logger.warning(f"Could not detect default branch via API: {e}. Trying fallbacks.")

            # Fallback: Check if 'master' is used if 'main' fails
            return "master"

    def get_repo_structure(self) -> list[dict]:
        """
        Fetches the entire file tree of the repository recursively.
        """
        session = self._ensure_session()
        default_branch = self._get_default_branch()

        try:
            # GitHub Git Database API: Get Tree recursively
            tree_url = f"{self.api_base}/git/trees/{default_branch}?recursive=1"
            response = session.get(tree_url)
            response.raise_for_status()

            return response.json().get("tree", [])
        except Exception as e:
            logger.error(f"Failed to retrieve repo structure: {e}")
            return []

    def identify_and_download_key_files(self) -> dict[str, list[Path]]:
        """
        Identifies key files (requirements, examples, docs) and downloads them to disk.
        Returns a dictionary mapping categories to local file paths.
        """
        # get_repo_structure calls _ensure_session internally, so we are safe here.
        structure = self.get_repo_structure()
        downloaded_files = {"requirements": [], "examples": [], "docs": []}

        to_download = []

        for item in structure:
            if item["type"] != "blob":
                continue  # Skip directories/submodules

            path_str = item["path"]
            filename = Path(path_str).name.lower()
            path_lower = path_str.lower()

            # Logic to categorize files
            category = None

            # 1. Dependency definitions
            if filename in ["requirements.txt", "setup.py", "pipfile", "pyproject.toml"]:
                category = "requirements"

            # 2. Examples or Demos (must be python files)
            elif (
                "example" in path_lower
                or "demo" in path_lower
                or "sample" in path_lower
                or "sample_script" in path_lower
            ) and path_str.endswith(".py"):
                category = "examples"

            # 3. Documentation
            elif "readme" in filename:
                category = "docs"

            if category:
                to_download.append((path_str, category))

        # Download the identified files
        logger.info(f"Found {len(to_download)} relevant files. Starting download...")
        for remote_path, category in to_download:
            local_path = self._download_file(remote_path)
            if local_path:
                downloaded_files[category].append(local_path)

        return downloaded_files

    def _download_file(self, remote_path: str) -> Path | None:
        """
        Downloads a specific file's content from the GitHub API, decodes it,
        and saves it to the local file system.
        """
        session = self._ensure_session()

        url = f"{self.api_base}/contents/{remote_path}"
        try:
            resp = session.get(url)
            resp.raise_for_status()
            data = resp.json()

            # GitHub API returns content in Base64
            if "content" not in data:
                return None

            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")

            # Construct local path
            local_file_path = self.save_dir / remote_path
            # Ensure subdirectories exist locally (e.g., examples/advanced/test.py)
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return local_file_path

        except Exception as e:
            logger.warning(f"Failed to download file ({remote_path}): {e}")
            return None
