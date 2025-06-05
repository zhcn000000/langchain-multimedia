import base64
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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
        input_image_url = None
        if not isinstance(message, str):
            for block in message:
                if not isinstance(block, str) and block.get("type") == "image_url":
                    input_image_url = block.get("image_url", {}).get("url")
                    break
        payload = {"prompt": prompt, **self.model_kwargs, **kwargs}
        if input_image_url is None:
            endpoint = self.endpoint + "/txt2img"
        else:
            input_image = urlopen(input_image_url).read()
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
        img_b64 = images[0]
        img_bytes = base64.b64decode(img_b64)
        cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = cache_dir / f"{uuid.uuid4()}.png"
        filename.write_bytes(img_bytes)

        # Return AIMessage with local file URL
        image_url = f"file://{filename}"
        return AIMessage(content=[{"type": "image_url", "image_url": {"url": image_url}}])
