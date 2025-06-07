import base64
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_multimedia.utils.helpers import _build_image, _find_image


class SDAPITextToImage(BaseChatModel):
    """Generate images from text prompts using Stable Diffusion REST API."""

    server_url: str
    """Base URL of the Stable Diffusion API, e.g. http://127.0.0.1:7860"""

    endpoint: str
    """API endpoint for text-to-image, default '/sdapi/v1/txt2img'"""

    model_kwargs: Dict[str, Any]
    """Additional payload parameters for the API call"""

    def __init__(
        self,
        server_url: str,
        endpoint: str = "/sdapi/v1",
        **model_kwargs: Any,
    ):
        super().__init__({"server_url": server_url, "endpoint": endpoint, "model_kwargs": model_kwargs})
        self.server_url = server_url.rstrip("/")
        self.endpoint = endpoint
        self.model_kwargs = model_kwargs or {}

    @property
    def _llm_type(self) -> str:
        return "sdapi-text-to-image"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"server_url": self.server_url, "endpoint": self.endpoint}

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

    def _convert_text_to_image(self, message: BaseMessage, **kwargs: Any) -> AIMessage:
        # Prepare request payload
        prompt = message.text()
        message = message.content
        input_image = _find_image(message)
        payload = {"prompt": prompt, **self.model_kwargs, **kwargs}
        if input_image is None:
            endpoint = self.endpoint + "/txt2img"
        else:
            payload["init_images"] = [base64.b64encode(input_image).decode("utf-8")]
            endpoint = self.endpoint + "/img2img"
        url = f"{self.server_url}{endpoint}"
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Expecting 'images' as list of base64-encoded strings
        images = data.get("images", [])
        if not images:
            raise ValueError("No images returned from Stable Diffusion API")

        # Decode first image and save to temp file
        content = []
        for image in images:
            image_data = base64.b64decode(image)
            content.append(_build_image(image_data))
        return AIMessage(content=content)
