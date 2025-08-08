import tempfile
from mimetypes import guess_extension
from pathlib import Path
from typing import Any, Optional, Union, Mapping
from uuid import uuid4

import requests
from langchain_community.utils.openai import is_openai_v1
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from magic import from_buffer
from pydantic import model_validator, Field, SecretStr
from anyio import Path as AsyncPath

cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
cache_dir.mkdir(parents=True, exist_ok=True)


class OpenAITool(BaseTool):
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default=None, alias="model")
    """Model name to use."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = Field(default_factory=from_env("OPENAI_PROXY", default=None))
    request_timeout: Union[float, tuple[float, float], Any, None] = Field(default=100, alias="timeout")
    max_retries: Optional[int] = Field(default=3, alias="max_retries")
    """Maximum number of retries to make when generating."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    http_client: Union[Any, None] = Field(default=None, exclude=True)
    """Optional httpx.Client. Only used for sync invocations. Must specify 
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = Field(default=None, exclude=True)
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify 
        http_client as well if you'd like a custom client for sync invocations."""

    @model_validator(mode="after")
    def validate_environment(self):
        """Validate that api key and python package exists in environment."""
        try:
            import openai

        except ImportError:
            raise ImportError("Could not import openai python package. Please install it with `pip install openai`.")

        if is_openai_v1():
            client_params = {
                "api_key": (self.openai_api_key.get_secret_value() if self.openai_api_key else None),
                "organization": self.openai_organization,
                "base_url": self.openai_api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
                "http_client": self.http_client,
            }

            if not self.client:
                self.client = openai.OpenAI(**client_params)
            if not self.async_client:
                self.async_client = openai.AsyncOpenAI(**client_params)
        elif not self.client:
            self.client = openai
        else:
            pass
        return self


class OpenAIAudioGenerator(OpenAITool):
    """OpenAIAudioGenerator is a tool that uses OpenAI to generate audio from text prompts."""

    name: str = "OpenAIAudioGenerator"
    description: str = "A tool that generates audio from text prompts using OpenAI's audio generation capabilities."
    voice: Optional[str] = Field(
        default=None,
        description="The voice to use for audio generation. If not provided, a default voice will be used.",
    )

    args_schema = {
        "prompt": {
            "type": "string",
            "description": "The text prompt to generate audio from"
        }
    }

    @staticmethod
    def _build_audio(output_audio):
        mime = from_buffer(output_audio, mime=True)
        ext = guess_extension(mime)
        filename = cache_dir / f"{uuid4()}{ext}"
        filename.write_bytes(output_audio)
        return filename

    @staticmethod
    async def _abuild_audio(output_audio):
        mime = from_buffer(output_audio, mime=True)
        ext = guess_extension(mime)
        filename = AsyncPath(cache_dir) / f"{uuid4()}{ext}"
        await filename.write_bytes(output_audio)
        return filename

    def _run(self, prompt: str) -> str:
        audio = self.client.audio.speech.create(
            model=self.model_name,
            input=prompt,
            voice=self.voice,
            **self.model_kwargs,
        ).content
        return str(self._build_audio(audio))

    async def _arun(self, prompt: str) -> str:
        audio = await self.async_client.audio.speech.create(
            model=self.model_name,
            input=prompt,
            voice=self.voice,
            **self.model_kwargs,
        )
        return str(await self._abuild_audio(audio.content))


class OpenAIImageGenerator(OpenAITool):
    """OpenAIImageGenerator is a tool that uses OpenAI to generate images from text prompts."""

    name: str = "OpenAIImageGenerator"
    description: str = "A tool that generates images from text prompts using OpenAI's image generation capabilities."

    args_schema = {
        "prompt": {
            "type": "string",
            "description": "The text prompt to generate image from"
        }
    }

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

    def _run(self, prompt: str, image: Optional[str]=None) -> str:
        """Run query through OpenAI and parse result."""
        if image is None:
            response = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                **self.model_kwargs,
            )
        else:
            image = Path(image).read_bytes()
            response = self.client.images.edit(
                model=self.model_name,
                prompt=prompt,
                image=image,
                **self.model_kwargs,
            )
        response = response.data[0].url
        response = requests.get(response).content
        return str(self._build_image(response))

    async def _arun(self, prompt: str, image: Optional[str]=None) -> str:
        """Run query through OpenAI async and parse result."""
        if image is None:
            response = await self.async_client.images.generate(
                model=self.model_name,
                prompt=prompt,
                **self.model_kwargs,
            )
        else:
            image_data = await AsyncPath(image).read_bytes()
            response = await self.async_client.images.edit(
                model=self.model_name,
                prompt=prompt,
                image=image_data,
                **self.model_kwargs,
            )
        response = response.data[0].url
        response = requests.get(response).content
        return str(await self._abuild_image(response))


class OpenAITranscriber(OpenAITool):
    """OpenAITranscriber is a tool that uses OpenAI to transcribe audio."""

    translation: bool = False
    name: str = "OpenAITranscriber"
    description: str = "A tool that transcribes audio using OpenAI's transcription capabilities."

    args_schema = {
        "audio": {
            "type": "string",
            "description": "Path to the audio file to transcribe"
        }
    }

    def _run(self, audio: str) -> str:
        """Run query through OpenAI and parse result."""
        audio_data = Path(audio).read_bytes()
        if self.translation:
            response = self.client.audio.translations.create(
                model=self.model_name,
                file=audio_data,
                **self.model_kwargs,
            )
        else:
            response = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_data,
                **self.model_kwargs,
            )
        return response.text

    async def _arun(self, audio: str) -> str:
        """Run query through OpenAI async and parse result."""
        audio_data = await AsyncPath(audio).read_bytes()
        if self.translation:
            response = await self.async_client.audio.translations.create(
                model=self.model_name,
                file=audio_data,
                **self.model_kwargs,
            )
        else:
            response = await self.async_client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_data,
                **self.model_kwargs,
            )
        return response.text
