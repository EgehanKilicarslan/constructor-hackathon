import json
import logging
import operator
import re
import shutil
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from service import ArticleAnalyser, ChatModel, GithubAnalyser

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ResultModel(BaseModel):
    project_name: str
    description: str
    files: dict[str, dict[str, str]]


class AgentState(TypedDict, total=False):
    article_url: str
    github_links: list[str]
    file_paths: Annotated[list[Path], operator.add]
    ai_response: str
    clean_response: str
    create_files: dict[str, str]


def _sanitize_filename(name: str) -> str:
    """Make filename safe for writing to disk."""
    name = name.strip()
    # Replace characters that are problematic on most filesystems
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    # Prevent path traversal
    name = Path(name).name
    return name or "file"


def download_and_analyze_article(state: AgentState) -> Dict[str, Any]:
    """
    Download the article and extract GitHub links.

    Returns a dict with keys 'github_links' and 'file_paths'.
    """
    url = state.get("article_url")
    if not url:
        logger.error("No article_url provided in state.")
        return {"github_links": [], "file_paths": []}

    # If the provided URL is a direct GitHub link, skip article download
    gh_match = re.match(r"^https?://(?:www\.)?github\.com/[^ \n]+", url, re.IGNORECASE)
    if gh_match:
        logger.info("Detected direct GitHub URL: %s", url)
        return {"github_links": [url], "file_paths": []}

    logger.info("Downloading article: %s", url)
    try:
        analyser = ArticleAnalyser(url=url)
        analyser.download()
    except Exception as e:
        logger.exception("Failed to download article %s: %s", url, e)
        return {"github_links": [], "file_paths": []}

    try:
        links = analyser.analyze_github_links() or []
    except Exception as e:
        logger.exception("Failed to analyze article for links: %s", e)
        links = []

    file_path = getattr(analyser, "file_path", None)
    file_paths: List[Path] = [Path(file_path)] if file_path else []

    if not links:
        logger.warning("No GitHub links found in article %s", url)

    return {"github_links": list(links), "file_paths": file_paths}


def process_github_repos(state: AgentState) -> Dict[str, List[Path]]:
    """
    Inspect found GitHub links and download key files.

    Returns dict with key 'file_paths' (list of Path).
    """
    links = state.get("github_links") or []
    new_files: List[Path] = []

    if not links:
        logger.info("No GitHub links to process.")
        return {"file_paths": new_files}

    for link in links:
        logger.info("Processing GitHub repo: %s", link)
        try:
            gh_analyser = GithubAnalyser(repo_url=link)
            gh_analyser.start()
            files_map = gh_analyser.identify_and_download_key_files() or {}
        except Exception as e:
            logger.exception("Error processing repo %s: %s", link, e)
            continue

        for category, paths in files_map.items():
            for p in paths:
                try:
                    new_files.append(Path(p))
                except Exception:
                    logger.warning("Skipping invalid path from repo %s: %s", link, p)

    return {"file_paths": new_files}


def generate_solution(state: AgentState) -> Dict[str, str]:
    """
    Upload collected files to the ChatModel and ask it to generate the demo project JSON.
    """
    logger.info("Generating solution via ChatModel.")
    system_prompt = """
    **Role:** You are a Senior Python Developer and DevOps Engineer.

    **Task:**
    1. Analyze the user's provided data or requirements description.
    2. Create a fully functional **Python demo script** to process or visualize this data.
    3. Create a **requirements.txt** file for dependencies.
    4. Create a **Dockerfile** to containerize the application.
    5. Create a comprehensive **README.md** explaining the project.

    **DevOps & Coding Standards (STRICTLY FOLLOW):**
    * **Python Version:** Do NOT use old versions like `3.10-slim-buster`. You MUST use **`python:3.11-slim`** (or newer) as the base image to ensure compatibility with modern libraries.
    * **Docker Copy Logic:** When copying configuration files that might be optional (specifically `.env`), you MUST use the wildcard syntax to prevent build failures if the file is missing.
        * *Incorrect:* `COPY .env ./`
        * *Correct:* `COPY .env* ./`
    * **Formatting:** Ensure the code is clean, PEP8 compliant, and well-commented.

    **JSON & Output Safety Rules (CRITICAL):**
    * **JSON Validity:** You must output **ONLY** a valid, raw JSON object. Do not include markdown formatting.
    * **Escaping (THE MOST IMPORTANT RULE):** Since the JSON values contain Python code:
        * You **MUST** properly escape all double quotes (`"`) inside the code as `\\"`.
        * You **MUST** replace all actual newlines with the escape sequence `\\n`.
        * **Comments & Docstrings:** Be extremely careful with Python comments (`#`) and docstrings (`\"\"\"`). Ensure they are strictly strictly compacted into the single-line JSON string format.
            * *Bad Example:* `"content": "def func(): \\n    \"\"\" doc \"\"\" "` (This breaks JSON if not escaped)
            * *Good Example:* `"content": "def func():\\n    \\\"\\\"\\\" doc \\\"\\\"\\\""` (This works)

    **JSON Template:**
    {
    "project_name": "demo_project",
    "description": "Brief summary of what the code does",
    "files": {
        "main.py": {
        "content": "INPUT_PYTHON_CODE_HERE"
        },
        "requirements.txt": {
        "content": "INPUT_REQUIREMENTS_HERE"
        },
        "Dockerfile": {
        "content": "INPUT_DOCKERFILE_CONTENT_HERE"
        },
        "README.md": {
        "content": "INPUT_MARKDOWN_CONTENT_HERE"
        }
    }
    }

    **User Input / Data:**
    """

    model = ChatModel(system_prompt=system_prompt)

    all_files = state.get("file_paths") or []
    for file_path in all_files:
        try:
            model.add_document(file_path)
        except Exception as e:
            logger.exception("Failed to add document %s to model: %s", file_path, e)

    prompt = (
        "Analyze the provided file context and data structure. Based on this, generate the full Python "
        "demo project, Dockerfile, and README as defined in the system instructions. Ensure the output is "
        "strictly the requested JSON format."
    )

    try:
        response = model.send(prompt)
    except Exception as e:
        logger.exception("ChatModel.send failed: %s", e)
        raise

    if not response:
        raise ValueError("Response from ChatModel is empty.")

    downloaded_files = Path("tmp/")
    if downloaded_files.exists():
        try:
            if downloaded_files.is_dir():
                shutil.rmtree(downloaded_files)
            else:
                downloaded_files.unlink()
        except Exception as e:
            logger.warning("Failed to remove tmp path %s: %s", downloaded_files, e)

    # Ensure response is a string
    return {"ai_response": str(response)}


def clean_ai_response(state: AgentState) -> Dict[str, str]:
    """
    Clean unwanted markdown fences and validate that the response is JSON-like.
    """
    raw = state.get("ai_response", "")
    content = raw.strip()

    # Remove common code fences
    if content.startswith("```"):
        # remove leading fence and optional language spec
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
        else:
            content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # If the model returned a pretty JSON with single quotes, try to normalize to valid JSON
    # but do not attempt destructive fixes; prefer to pass through and let validation fail explicitly.
    return {"clean_response": content}


def create_files_from_response(state: AgentState) -> Dict[str, str]:
    """
    Parse the cleaned AI response (expected JSON) and create files on disk.

    Returns mapping of created filenames to their absolute paths.
    """
    content = state.get("clean_response", "")
    if not content:
        raise ValueError("No cleaned AI response available to create files.")

    # Try to ensure it's valid JSON first for clearer errors
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.exception("AI response is not valid JSON: %s", e)
        # Try Pydantic validation which will give clearer messages if it's a valid JSON string representation
        try:
            result = ResultModel.model_validate_json(content)
        except ValidationError as e2:
            logger.error("Validation error for AI response: %s", e2)
            raise ValueError("AI response is not in the expected ResultModel format.") from e2
    else:
        try:
            result = ResultModel.model_validate(parsed)
        except ValidationError as e:
            logger.error("AI response structure invalid: %s", e)
            raise ValueError("AI response is not in the expected ResultModel format.") from e

    files_created: Dict[str, str] = {}
    base_dir = Path("results") / result.project_name
    base_dir.mkdir(parents=True, exist_ok=True)

    for filename, fileinfo in result.files.items():
        safe_name = _sanitize_filename(filename)
        file_path = (base_dir / safe_name).resolve()
        content_text = ""
        if isinstance(fileinfo, dict):
            content_text = fileinfo.get("content", "")
        elif isinstance(fileinfo, str):
            content_text = fileinfo
        else:
            logger.warning("Unexpected fileinfo type for %s: %s", filename, type(fileinfo))
            content_text = str(fileinfo)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                f.write(content_text)
            files_created[filename] = str(file_path)
            logger.info("Created file %s", file_path)
        except Exception as e:
            logger.exception("Failed to write file %s: %s", file_path, e)

    return {"create_files": json.dumps(files_created)}


# 3. BUild the graph (Workflow)

workflow = StateGraph(AgentState)

# Add nodes (States)
workflow.add_node("extract_article", download_and_analyze_article)
workflow.add_node("process_repos", process_github_repos)
workflow.add_node("generate", generate_solution)
workflow.add_node("clean", clean_ai_response)
workflow.add_node("create_files", create_files_from_response)

# Add edges (transitions)
workflow.set_entry_point("extract_article")
workflow.add_edge("extract_article", "process_repos")
workflow.add_edge("process_repos", "generate")
workflow.add_edge("generate", "clean")
workflow.add_edge("clean", "create_files")
workflow.add_edge("create_files", END)

# Compile the graph into an application
app = workflow.compile()
