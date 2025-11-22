from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import HttpUrl

from adapter import StatelessConstructorAdapter


class ConstructorModel(ChatOpenAI):
    def __init__(
        self,
        adapter: StatelessConstructorAdapter | None = None,
        api_url: HttpUrl | None = None,
        api_key: str | None = None,
        km_id: str | None = None,
        llm_name: str | None = None,
        llm_alias: str | None = "gpt-5-pro",
        **kwargs,
    ) -> None:
        if adapter is None:
            self.adapter = StatelessConstructorAdapter(
                api_url=api_url,
                api_key=api_key,
                km_id=km_id,
                llm_name=llm_name,
                llm_alias=llm_alias,
            )
        else:
            self.adapter = adapter

        kwargs["api_key"] = "unused"
        kwargs["base_url"] = f"{self.adapter.api_url}/knowledge-models/{self.adapter.km_id}"
        kwargs["model"] = self.adapter.llm_alias

        super().__init__(**kwargs)

    def _get_request_payload(self, *args, **kwargs) -> dict[Any, Any]:
        res = super()._get_request_payload(*args, **kwargs)

        base_headers = res.get("extra_headers", {})
        adapter_headers = self.adapter._get_headers()
        base_headers.update(adapter_headers)
        base_headers["X-KM-Extension"] = "direct_llm"

        res["extra_headers"] = base_headers
        return res
