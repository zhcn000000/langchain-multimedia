from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_multimedia.utils.helpers import _build_video, _find_image
from langchain_multimedia.core import BaseGenerationModel, GenerationType

class XinferenceVideoGenerator(BaseGenerationModel):
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

    def _check_cluster_authenticated(self) -> None:
        url = f"{self.server_url}/v1/cluster/auth"
        response = requests.get(url)
        if response.status_code == 404:
            self._cluster_authed = False
        else:
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get cluster information, detail: {response.json()['detail']}")
            response_data = response.json()
            self._cluster_authed = bool(response_data["auth"])

    @property
    def _generator_type(self) -> GenerationType:
        """Return the type of generator."""
        return GenerationType.VideoGenerator

    @property
    def _llm_type(self) -> str:
        return "xinference-text-to-video"

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
            ai_msg = self._convert_text_to_video(message)
            gen = ChatGeneration(message=ai_msg)
            results.append(gen)
        return ChatResult(generations=results)

    def _convert_text_to_video(self, message: BaseMessage) -> AIMessage:
        prompt = message.text()
        model = self.client.get_model(self.model_uid)
        input_image = _find_image(message)
        if not prompt:
            raise ValueError("Prompt is required for image generation.")
        if input_image is not None:
            vdo = model.image_to_video(prompt=prompt, image=input_image)
        else:
            vdo = model.text_to_video(prompt=prompt)
        video = open(vdo["data"]["url"], "rb").read()
        return AIMessage(content=[_build_video(video)])
