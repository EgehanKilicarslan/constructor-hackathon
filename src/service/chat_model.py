from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from adapter import StatelessConstructorAdapter


class ConstructorModel(ChatOpenAI):
    _adapter: StatelessConstructorAdapter = PrivateAttr()

    def __init__(
        self,
        adapter: StatelessConstructorAdapter | None = None,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> None:
        if adapter is None:
            _adapter = StatelessConstructorAdapter(llm_alias=model)
        else:
            _adapter = adapter

        kwargs["api_key"] = "unused"
        kwargs["base_url"] = f"{_adapter.api_url}/knowledge-models/{_adapter.km_id}"
        kwargs["model"] = _adapter.llm_alias

        super().__init__(**kwargs)
        self._adapter = _adapter

    def _get_request_payload(self, *args, **kwargs) -> dict[Any, Any]:
        res = super()._get_request_payload(*args, **kwargs)
        res["extra_headers"] = self._adapter._get_headers()
        res["extra_headers"]["X-KM-Extension"] = "direct_llm"

        return res

    def send(self, human: str, system: str | None = None) -> str | list[str | dict[Any, Any]]:
        _system = SystemMessage(content=system)
        _human = HumanMessage(content=human)

        messages = [_system, _human] if _system is not None else [_human]
        return super().invoke(messages).content

    # --- new methods: document helpers that delegate to the adapter ---
    def add_document(self, file_path: str) -> dict | None:
        """
        Upload a file to the Constructor knowledge model via the adapter.
        Returns the adapter's JSON response or raises on error.
        """
        return self._adapter.add_document(file_path)

    def add_facts(self, content: dict) -> dict | None:
        """
        Add key/value facts to the KM by creating a temporary markdown and
        uploading it via the adapter (uses Adapter.add_facts).
        """
        return self._adapter.add_facts(content)
