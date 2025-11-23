import requests
from pydantic import HttpUrl

from .adapter_base import ConstructorAdapter


class StatelessConstructorAdapter(ConstructorAdapter):
    """
    Stateless Adapter for Constructor API
    """

    def __init__(
        self,
        api_url: HttpUrl | None = None,
        api_key: str | None = None,
        km_id: str | None = None,
        llm_name: str | None = None,
        llm_alias: str = "gpt-4o-mini",
    ):
        super().__init__(api_url, api_key, km_id, llm_name, llm_alias)

    def query(self, question: str) -> str:
        endpoint = f"{self.api_url}/knowledge-models/{self.km_id}/chat/completions"

        payload = {
            "messages": [{"role": "user", "content": question}],
            "mode": "model",
            "model": self.llm_alias,
        }

        response = requests.post(endpoint, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices and "message" in choices[0] and "content" in choices[0]["message"]:
            return choices[0]["message"]["content"]
        else:
            return "No response received from the model."
