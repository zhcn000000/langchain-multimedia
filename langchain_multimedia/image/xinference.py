import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Any, Dict, Mapping
from urllib.request import urlopen

import magic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import requests


class XinferenceTextToImage(BaseChatModel):
    """XinferenceImage is a chat model that uses xinference to generate images from text prompts or image inputs."""

    client: Optional[Any] = None
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""
    model_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to xinference.LLM"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None,
        api_key: Optional[str] = None,
        **model_kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            try:
                from xinference_client import RESTfulClient
            except ImportError as e:
                raise ImportError(
                    "Could not import RESTfulClient from xinference. Please install it"
                    " with `pip install xinference` or `pip install xinference_client`."
                ) from e

        model_kwargs = model_kwargs or {}

        super().__init__(
            **{  # type: ignore[arg-type]
                "server_url": server_url,
                "model_uid": model_uid,
                "model_kwargs": model_kwargs,
            }
        )

        if self.server_url is None:
            raise ValueError("Please provide server URL")

        if self.model_uid is None:
            raise ValueError("Please provide the model UID")

        self._headers: Dict[str, str] = {}
        self._cluster_authed = False
        self._check_cluster_authenticated()
        if api_key is not None and self._cluster_authed:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self.client = RESTfulClient(server_url, api_key)

    @property
    def _llm_type(self) -> str:
        return "xinference-text-to-image"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"server_url": self.server_url},
            **{"model_uid": self.model_uid},
            **{"model_kwargs": self.model_kwargs},
        }


    def _check_cluster_authenticated(self) -> None:
        url = f"{self.server_url}/v1/cluster/auth"
        response = requests.get(url)
        if response.status_code == 404:
            self._cluster_authed = False
        else:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get cluster information, "
                    f"detail: {response.json()['detail']}"
                )
            response_data = response.json()
            self._cluster_authed = bool(response_data["auth"])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        results = []
        for message in messages:
            ai_msg = self._convert_text_to_image(message)
            gen = ChatGeneration(message=ai_msg)
            results.append(gen)
        return ChatResult(generations=results)


    def _convert_text_to_image(self, message: BaseMessage, **kwargs) -> AIMessage:
        message = message
        model = self.client.get_model(self.model_uid)
        prompt = message.text()
        input_image_url = None
        if not isinstance(message, str):
            for block in message.content:
                if not isinstance(block, str) and block.get("type") == "image_url":
                    input_image_url = block.get("image_url", {}).get("url")
                    break
        if not prompt:
            raise ValueError("Prompt is required for image generation.")
        if input_image_url is not None:
            input_image = urlopen(input_image_url).read()
            response = model.image_to_image(prompt=prompt, image=input_image, **kwargs)
        else:
            response = model.text_to_image(prompt=prompt, **kwargs)

        output_image_url = response["data"]["url"]
        if output_image_url.startswith("/"):
            output_image_url = "file://" + output_image_url

        image = urlopen(output_image_url).read()
        cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
        cache_dir.mkdir(parents=True, exist_ok=True)
        mime = magic.from_buffer(image, mime=True)
        ext = mime.split("/")[-1]
        filename = cache_dir / f"{uuid.uuid4()}.{ext}"
        with open(filename, "wb") as f:
            f.write(image)
        image_url = f"file://{filename}"
        return AIMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
            ]
        )


class XinferenceImageToText(BaseChatModel):
    """XinferenceImage is a chat model that uses xinference to generate images from text prompts or image inputs."""

    client: Optional[Any] = None
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""
    model_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to xinference.LLM"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None,
        api_key: Optional[str] = None,
        **model_kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            try:
                from xinference_client import RESTfulClient
            except ImportError as e:
                raise ImportError(
                    "Could not import RESTfulClient from xinference. Please install it"
                    " with `pip install xinference` or `pip install xinference_client`."
                ) from e

        model_kwargs = model_kwargs or {}

        super().__init__(
            **{  # type: ignore[arg-type]
                "server_url": server_url,
                "model_uid": model_uid,
                "model_kwargs": model_kwargs,
            }
        )

        if self.server_url is None:
            raise ValueError("Please provide server URL")

        if self.model_uid is None:
            raise ValueError("Please provide the model UID")

        self._headers: Dict[str, str] = {}
        self._cluster_authed = False
        self._check_cluster_authenticated()
        if api_key is not None and self._cluster_authed:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self.client = RESTfulClient(server_url, api_key)

    @property
    def _llm_type(self) -> str:
        return "xinference-image-to-text"

    def _check_cluster_authenticated(self) -> None:
        url = f"{self.server_url}/v1/cluster/auth"
        response = requests.get(url)
        if response.status_code == 404:
            self._cluster_authed = False
        else:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get cluster information, "
                    f"detail: {response.json()['detail']}"
                )
            response_data = response.json()
            self._cluster_authed = bool(response_data["auth"])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"server_url": self.server_url},
            **{"model_uid": self.model_uid},
            **{"model_kwargs": self.model_kwargs},
        }


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        results = []
        for message in messages:
            ai_msg = self._convert_image_to_text(message)
            gen = ChatGeneration(message=ai_msg)
            results.append(gen)
        return ChatResult(generations=results)

    def _convert_image_to_text(self, message: BaseMessage, **kwargs) -> AIMessage:
        image_url = None
        message = message.content
        model = self.client.get_model(self.model_uid)
        if isinstance(message, str):
            raise ValueError("Image is required for generate text.")
        for block in message:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image_url = block.get("image_url", {}).get("url")
                break
        if not image_url:
            raise ValueError("Prompt is required for image generation.")
        image = urlopen(image_url).read()
        text = model.ocr(image=image, **kwargs)
        return AIMessage(
            content=[
                {
                    "type": "text",
                    "text": text,
                }
            ]
        )
