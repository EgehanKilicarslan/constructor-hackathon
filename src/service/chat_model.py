import logging
import os
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UploadedFile(BaseModel):
    filename: str
    path: Path
    content: str | bytes


class ChatModel:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,  # For file analysis, lower temperature for more accurate answers
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self._llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
        )
        self._system_prompt = system_prompt
        self._uploaded_files: list[UploadedFile] = []

    def add_document(self, file: Path) -> None:
        """Dosyayı yükler ve içeriğini okur."""
        if not file.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file}")

        with file.open("rb") as f:
            content = f.read()

        extracted_text = ""
        if file.suffix.lower() == ".pdf":
            extracted_text = self._parse_pdf(content)
        elif file.suffix.lower() in [".txt", ".md", ".py", ".csv"]:
            extracted_text = content.decode("utf-8", errors="ignore")

        uploaded_file = UploadedFile(filename=file.name, path=file, content=extracted_text)
        self._uploaded_files.append(uploaded_file)
        logger.debug(f"Uploaded file: {uploaded_file}")

    def add_documents(self, files: list[Path]) -> None:
        for file in files:
            self.add_document(file)

    def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += str(page.get_text()) + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file: {e}")

    def send(self, user_message: str) -> str | list[str | dict[Any, Any]] | None:
        file_context = ""
        if self._uploaded_files:
            for file in self._uploaded_files:
                file_context += f"\nFilename: {file.filename}\nFilepath: {file.path.as_posix()}\nContent:\n{file.content}\n"

        messages = [
            SystemMessage(content=self._system_prompt + "\n\n" + file_context),
            HumanMessage(content=user_message),
        ]

        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
