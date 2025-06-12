import base64
import tempfile
from mimetypes import guess_extension
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import requests
from langchain_core.tools import BaseTool
from magic import from_buffer

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

    def _convert_text_to_image(self, prompt: str, image: Optional[str], **kwargs: Any) -> str:
        # Prepare request payload
        payload = {"prompt": prompt, **self.model_kwargs, **kwargs}
        if image is None:
            endpoint = self.endpoint + "/txt2img"
        else:
            image = Path(image).read_bytes()
            payload["init_images"] = [base64.b64encode(image).decode("utf-8")]
            endpoint = self.endpoint + "/img2img"
        url = f"{self.server_url}{endpoint}"
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Expecting 'images' as list of base64-encoded strings
        response = data.get("images", [])
        if not response:
            raise ValueError("No images returned from Stable Diffusion API")
        response = response[0]
        image_data = base64.b64decode(response)
        return str(self._build_image(image_data))
