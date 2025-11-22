import logging
import os
import tempfile
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, ValidationError

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

AI_MESSAGE_TYPE = "ai_message"
PROCESSING_STATUS = "processing"
DONE_STATUS = "done"
KNOWLEDGE_MODEL_ONLY = "model"
LLM_ENGINE_ONLY = "direct"


class AdapterSettings(BaseModel):
    api_url: HttpUrl = Field(
        default=HttpUrl("https://training.constructor.app/api/platform-kmapi"),
        description="Constructor KM API base URL",
    )
    api_key: str = Field(min_length=1)
    km_id: str = Field(min_length=1)
    llm_name: str | None = None
    llm_alias: str = "gpt-5-pro"

    @classmethod
    def from_env(
        cls,
        api_url: HttpUrl | None = None,
        api_key: str | None = None,
        km_id: str | None = None,
        llm_name: str | None = None,
        llm_alias: str | None = None,
    ) -> "AdapterSettings":
        load_dotenv()

        return cls(
            api_url=HttpUrl(
                api_url or os.getenv("CONSTRUCTOR_API_URL") or cls.model_fields["api_url"].default
            ),
            api_key=api_key or os.getenv("CONSTRUCTOR_API_KEY") or "",
            km_id=km_id or os.getenv("CONSTRUCTOR_KM_ID") or "",
            llm_name=llm_name,
            llm_alias=llm_alias or "gpt-5-pro",
        )


class LLMInfo(BaseModel):
    id: str
    name: str
    alias: str | None = None


class LLMListResponse(BaseModel):
    results: list[LLMInfo] = Field(default_factory=list)


class FileInfo(BaseModel):
    id: str
    filename: str


class FileListResponse(BaseModel):
    results: list[FileInfo] = Field(default_factory=list)


class ConstructorAdapter(ABC):
    """
    Abstract Base Class for Constructor Adapter
    """

    def __init__(
        self,
        api_url: HttpUrl | None = None,
        api_key: str | None = None,
        km_id: str | None = None,
        llm_name: str | None = None,
        llm_alias: str | None = "gpt-5-pro",
    ):
        try:
            self.settings = AdapterSettings.from_env(
                api_url=api_url,
                api_key=api_key,
                km_id=km_id,
                llm_name=llm_name,
                llm_alias=llm_alias,
            )
        except ValidationError as e:
            logging.error(f"Adapter configuration error: {e}")
            raise

        self.api_url = str(self.settings.api_url).rstrip("/")
        self.api_key = self.settings.api_key
        self.km_id = self.settings.km_id
        self.llm_name = self.settings.llm_name
        self.llm_alias = self.settings.llm_alias

        self.llms: dict[str, LLMInfo] = self._gather_llms()
        self.llm_id = self.get_llm_id(self.llm_alias)

        if self.llm_name is None:
            self.llm_name = self.get_llm_name(self.llm_alias)

    @abstractmethod
    def query(self, question: str, **kwargs) -> str:
        """
        Abstract method to query the Constructor Model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_headers(self) -> dict[str, str]:
        return {"X-KM-AccessKey": f"Bearer {self.api_key}"}

    # ---------- LLM METHODS ----------

    def get_available_llms(self) -> LLMListResponse:
        resp = requests.get(f"{self.api_url}/language_models", headers=self._get_headers())
        resp.raise_for_status()
        try:
            return LLMListResponse.model_validate(resp.json())
        except ValidationError as e:
            logging.error(f"Invalid LLM list response: {e}")
            raise

    def _gather_llms(self) -> dict[str, LLMInfo]:
        llms_response = self.get_available_llms()
        llms_map: dict[str, LLMInfo] = {}
        for llm in llms_response.results:
            alias = llm.alias or llm.name
            llms_map[alias] = llm
        return llms_map

    def get_llm_id(self, llm_alias: str) -> str:
        llm = self.llms.get(llm_alias)
        if llm is None:
            raise ValueError(f"LLM with alias '{llm_alias}' not found.")
        return llm.id

    def get_llm_name(self, llm_alias: str) -> str:
        llm = self.llms.get(llm_alias)
        if llm is None:
            raise ValueError(f"LLM with alias '{llm_alias}' not found.")
        return llm.name

    # ---------- FILE / KM METHODS ----------

    def add_document(self, file_path: str):
        if not os.path.isfile(file_path):
            msg = f"File '{file_path}' does not exist or is not accessible."
            logging.error(msg)
            return {"error": msg}

        endpoint = f"{self.api_url}/knowledge-models/{self.km_id}/files"
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(endpoint, headers=self._get_headers(), files=files)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error uploading file '{file_path}': {e}")
            raise

    def get_all_documents(self) -> list[FileInfo]:
        endpoint = f"{self.api_url}/knowledge-models/{self.km_id}/files"
        try:
            response = requests.get(endpoint, headers=self._get_headers())
            response.raise_for_status()
            file_list = FileListResponse.model_validate(response.json())
            return file_list.results
        except (requests.exceptions.RequestException, ValidationError) as e:
            logging.error(f"Error querying documents: {e}")
            raise

    def get_all_documents_names(self) -> list[str]:
        docs = self.get_all_documents()
        return [doc.filename for doc in docs]

    def delete_document_by_id(self, document_id: str) -> bool:
        endpoint = f"{self.api_url}/knowledge-models/{self.km_id}/files/{document_id}"
        try:
            response = requests.delete(endpoint, headers=self._get_headers())
            if response.status_code in (200, 204):
                logging.info(f"Document {document_id} deleted successfully.")
                return True
            logging.error(
                f"Failed to delete document {document_id}: {response.status_code}, {response.text}"
            )
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Error deleting document {document_id}: {e}")
            raise

    def delete_all_documents(self) -> str:
        documents = self.get_all_documents()
        if not documents:
            logging.info("No documents found to delete.")
            return "No documents found to delete."

        for document in documents:
            logging.info(f"Deleting document {document.id}...")
            self.delete_document_by_id(document.id)

        return "All documents deleted successfully."

    def reset_model(self) -> None:
        self.delete_all_documents()

    def delete_model(self) -> str:
        endpoint = f"{self.api_url}/knowledge-models/{self.km_id}"
        try:
            response = requests.delete(endpoint, headers=self._get_headers())
            if response.status_code in (200, 204):
                logging.info(f"Knowledge Model {self.km_id} reset successfully.")
                return "Knowledge Model reset successfully."
            logging.error(
                f"Failed to reset Knowledge Model: {response.status_code}, {response.text}"
            )
            return f"Failed to reset Knowledge Model: {response.status_code}"
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during Knowledge Model reset: {e}")
            raise

    def add_facts(self, content: dict):
        markdown_content = "\n".join([f"{key}: {value}" for key, value in content.items()])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
            temp_file.write(markdown_content)
            temp_filepath = temp_file.name

        try:
            return self.add_document(temp_filepath)
        finally:
            try:
                os.remove(temp_filepath)
            except OSError:
                pass
