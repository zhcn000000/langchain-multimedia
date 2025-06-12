import tempfile
from mimetypes import guess_extension
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import requests
from langchain_core.tools import BaseTool
from magic import from_buffer
from pydantic import Field

cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
cache_dir.mkdir(parents=True, exist_ok=True)


class XinferenceTool(BaseTool):
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


class XinferenceAudioGenerator(XinferenceTool):
    name: str = "XinferenceAudioGenerator"
    description: str = "A tool that generates audio from text prompts using Xinference's audio generation capabilities."
    voice: Optional[str] = Field(
        default=None,
        description="The voice to use for audio generation. If not provided, a default voice will be used.",
    )
    @staticmethod
    def _build_audio(output_audio):
        mime = from_buffer(output_audio, mime=True)
        ext = guess_extension(mime)
        filename = cache_dir / f"{uuid4()}{ext}"
        filename.write_bytes(output_audio)
        return filename

    def _run(self, prompt: str) -> str:
        model = self.client.get_model(self.model_uid)
        audio = model.speech(input=prompt,voice=self.voice, **self.model_kwargs)
        return str(self._build_audio(audio))


class XinferenceImageGenerator(XinferenceTool):
    name: str = "XinferenceImageGenerator"
    description: str = (
        "A tool that generates images from text prompts using Xinference's image generation capabilities."
    )

    @staticmethod
    def _build_image(output_image):
        mime = from_buffer(output_image, mime=True)
        ext = guess_extension(mime)
        filename = cache_dir / f"{uuid4()}{ext}"
        filename.write_bytes(output_image)
        return filename

    def _run(self, prompt: str, image: Optional[str]=None) -> str:
        model = self.client.get_model(self.model_uid)
        if image is None:
            response = model.text_to_image(prompt=prompt, **self.model_kwargs)
        else:
            image = Path(image).read_bytes()
            response = model.image_to_image(prompt=prompt, image=image, **self.model_kwargs)
        return str(self._build_image(response))


class XinferenceVideoGenerator(XinferenceTool):
    name: str = "XinferenceVideoGenerator"
    description: str = (
        "A tool that generates videos from text prompts using Xinference's video generation capabilities."
    )

    @staticmethod
    def _build_video(output_video):
        mime = from_buffer(output_video, mime=True)
        ext = guess_extension(mime)
        filename = cache_dir / f"{uuid4()}{ext}"
        filename.write_bytes(output_video)
        return str(filename)

    def _run(self, prompt: str, image: Optional[str]=None) -> str:
        model = self.client.get_model(self.model_uid)
        if image is None:
            response = model.text_to_video(prompt=prompt, **self.model_kwargs)
        else:
            image = Path(image).read_bytes()
            response = model.image_to_video(prompt=prompt, image=image, **self.model_kwargs)
        return str(self._build_video(response))


class XinferenceTranscriber(XinferenceTool):
    name: str = "XinferenceTranscriber"
    description: str = "A tool that transcribes audio files using Xinference's transcription capabilities."
    translation: bool = False

    def _run(self, audio: str) -> str:
        model = self.client.get_model(self.model_uid)
        audio_data = Path(audio).read_bytes()
        if self.translation:
            response = model.translations(audio=audio_data, **self.model_kwargs)
        else:
            response = model.transcriptions(audio=audio_data, **self.model_kwargs)
        return response.get("text", "")


class XinferenceOCR(XinferenceTool):
    name: str = "XinferenceOCR"
    description: str = "A tool that performs OCR on images using Xinference's OCR capabilities."

    def _run(self, image: str) -> str:
        model = self.client.get_model(self.model_uid)
        image_data = Path(image).read_bytes()
        response = model.ocr(image=image_data, **self.model_kwargs)
        return response.get("text", "")
