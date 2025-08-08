import base64
import tempfile
from mimetypes import guess_extension
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx

from langchain_core.tools import BaseTool
from magic import from_buffer
from anyio import Path as AsyncPath

cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"

cache_dir.mkdir(parents=True, exist_ok=True)


class StableDiffusionImageGenerator(BaseTool):
    server_url: str
    """Base URL of the Stable Diffusion API, e.g. http://127.0.0.1:7860"""

    endpoint: str
    """API endpoint for text-to-image, default '/sdapi/v1/txt2img'"""

    model_kwargs: Dict[str, Any]
    """Additional payload parameters for the API call"""
    name: str = "StableDiffusionImageGenerator"
    description: str = "Generates images from text prompts using Stable Diffusion API. "

    args_schema = {
        "prompt": {
            "type": "string",
            "description": "The text prompt to generate image from"
        }
    }

    def __init__(
        self,
        server_url: str,
        endpoint: str = "/sdapi/v1",
        **model_kwargs: Any,
    ):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.endpoint = endpoint
        self.model_kwargs = model_kwargs or {}

    @staticmethod
    def _build_image(output_image):
        mime = from_buffer(output_image, mime=True)
        ext = guess_extension(mime)
        filename = cache_dir / f"{uuid4()}{ext}"
        filename.write_bytes(output_image)
        return filename

    @staticmethod
    async def _abuild_image(output_image):
        mime = from_buffer(output_image, mime=True)
        ext = guess_extension(mime)
        filename = AsyncPath(cache_dir) / f"{uuid4()}{ext}"
        await filename.write_bytes(output_image)
        return filename

    def _convert_text_to_image(self, prompt: str, image: Optional[str], **kwargs: Any) -> str:
        # Prepare request payload
        payload = {"prompt": prompt, **self.model_kwargs, **kwargs}
        if image is None:
            endpoint = self.endpoint + "/txt2img"
        else:
            image_data = Path(image).read_bytes()
            payload["init_images"] = [base64.b64encode(image_data).decode("utf-8")]
            endpoint = self.endpoint + "/img2img"
        url = f"{self.server_url}{endpoint}"

        with httpx.Client() as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Expecting 'images' as list of base64-encoded strings
        response = data.get("images", [])
        if not response:
            raise ValueError("No images returned from Stable Diffusion API")
        response = response[0]
        image_data = base64.b64decode(response)
        return str(self._build_image(image_data))

    async def _aconvert_text_to_image(self, prompt: str, image: Optional[str], **kwargs: Any) -> str:
        # Prepare request payload
        payload = {"prompt": prompt, **self.model_kwargs, **kwargs}
        if image is None:
            endpoint = self.endpoint + "/txt2img"
        else:
            image_data = await AsyncPath(image).read_bytes()
            payload["init_images"] = [base64.b64encode(image_data).decode("utf-8")]
            endpoint = self.endpoint + "/img2img"
        url = f"{self.server_url}{endpoint}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Expecting 'images' as list of base64-encoded strings
        response = data.get("images", [])
        if not response:
            raise ValueError("No images returned from Stable Diffusion API")
        response = response[0]
        image_data = base64.b64decode(response)
        return str(await self._abuild_image(image_data))

    def _run(self, prompt: str, image: Optional[str] = None) -> str:
        return self._convert_text_to_image(prompt, image)

    async def _arun(self, prompt: str, image: Optional[str] = None) -> str:
        return await self._aconvert_text_to_image(prompt, image)
