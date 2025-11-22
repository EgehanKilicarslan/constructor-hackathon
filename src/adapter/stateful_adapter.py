import logging
import time

import requests
from pydantic import HttpUrl

from .adapter_base import (
    AI_MESSAGE_TYPE,
    DONE_STATUS,
    KNOWLEDGE_MODEL_ONLY,
    LLM_ENGINE_ONLY,
    PROCESSING_STATUS,
    ConstructorAdapter,
)


class StatefulConstructorAdapter(ConstructorAdapter):
    """
    Stateful Adapter for Constructor API
    """

    def __init__(
        self,
        default_model_mode=LLM_ENGINE_ONLY,
        api_url: HttpUrl | None = None,
        api_key: str | None = None,
        km_id: str | None = None,
        llm_name: str | None = None,
        llm_alias: str | None = "gpt-5-pro",
    ):
        super().__init__(api_url, api_key, km_id, llm_name, llm_alias)
        self.mode = default_model_mode  # used for selecting the mode of the model, LLM_ENGINE_ONLY (default) or KNOWLEDGE_MODEL_ONLY
        self.session_id = None
        self._start_session()

    def _start_session(self):
        if not self.session_id:
            data = {"llm_id": self.llm_id, "mode": self.mode}
            url = f"{self.api_url}/knowledge-models/{self.km_id}/chat-sessions"
            response = requests.post(url, headers=self._get_headers(), json=data)

            if response.status_code == 200:
                self.session_id = response.json().get("id")
                logging.info(f"Session started. Session ID: {self.session_id}")
            else:
                print(f"Failed to start session: {response.status_code}, {response.text}")
                raise Exception(f"Failed to start session: {response.status_code}, {response.text}")

    def restart_session(self):
        self._start_session()

    def query(
        self, question: str, timeout=120, request_timeout=15, retry_delay=3, mode=True
    ) -> str | None:
        """
        Sends a query to the AI model and retrieves its response.

        Args:
            question (str): The question to send to the AI model.
            timeout (int, optional): The maximum time (in seconds) to wait for a response. Defaults to 120.
            request_timeout (int, optional): The timeout (in seconds) for each HTTP request. Defaults to 15.
            retry_delay (int, optional): The delay (in seconds) between retries when waiting for a response. Defaults to 3.
            mode (bool, optional): The mode to use when sending the message. Defaults to True.

        Returns:
            str | None: The response text from the AI model, or None if no valid response is received.

        Raises:
            TimeoutError: If the response from the AI model exceeds the specified timeout.
        """
        self._send_message(question, mode)
        start_time = time.time()

        while True:
            response = requests.get(
                f"{self.api_url}/knowledge-models/{self.km_id}/chat-sessions/{self.session_id}/messages",
                headers=self._get_headers(),
                timeout=request_timeout,
            )
            message = response.json()["results"][0]
            if message["type"] != AI_MESSAGE_TYPE:
                break

            status_name = message["status"]["name"]
            if status_name == PROCESSING_STATUS:
                logging.info("Waiting for reply...")
                time.sleep(retry_delay)
            elif status_name == DONE_STATUS:
                messages = response.json().get("results", [])
                for message in messages:
                    if (
                        message["type"] == AI_MESSAGE_TYPE
                        and message["status"]["name"] == DONE_STATUS
                    ):
                        # Access the nested 'content' dictionary and then the 'text' field
                        content = message.get("content", {})
                        return content.get("text", "No response text available")
                return "Unclear answer" + message
            if time.time() - start_time > timeout:
                raise TimeoutError("Model response timed out.")

    def _send_message(self, message: str, mode: bool):
        """
        Sends a message to the specified chat session of the knowledge model.

        Args:
            message (str): The message text to be sent.
            mode (bool): The mode to use for sending the message. If True, the
                message is sent using the LLM engine only; if False, it is sent
                using the knowledge model only.

        Raises:
            HTTPError: If the HTTP request to send the message fails.
        """

        actual_mode = LLM_ENGINE_ONLY if mode else KNOWLEDGE_MODEL_ONLY
        data = {"text": message, "mode": actual_mode}
        response = requests.post(
            f"{self.api_url}/knowledge-models/{self.km_id}/chat-sessions/{self.session_id}/messages",
            headers=self._get_headers(),
            json=data,
        )
        response.raise_for_status()
