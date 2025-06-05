import tempfile
import uuid
from pathlib import Path
from urllib.request import urlopen
from typing import List, Any, Dict, Optional

import magic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from io import BytesIO


class OpenAITextToAudio(BaseChatOpenAI):
    """Generate audio from text prompts using OpenAI TTS API."""

    @property
    def _llm_type(self) -> str:
        return "openai-text-to-audio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        results = []
        for message in messages:
            ai_msg = self._convert_text_to_audio(message, **kwargs)
            gen = ChatGeneration(message=ai_msg)
            results.append(gen)
        return ChatResult(generations=results)

    def _convert_text_to_audio(self, message: BaseMessage, **kwargs) -> AIMessage:
        # Extract the first user message as text prompt
        prompt = message.text()

        # Call OpenAI TTS endpoint
        audio = self.root_client.audio.speech.create(model=self.model_name, input=prompt, **kwargs).content

        # Save audio to a temp file
        cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
        cache_dir.mkdir(parents=True, exist_ok=True)
        mime = magic.from_buffer(audio, mime=True)
        ext = mime.split("/")[-1]
        if ext == "x-wav":
            ext = "wav"
        elif ext == "mpeg":
            ext = "mp3"
        filename = cache_dir / f"{uuid.uuid4()}.{ext}"
        with open(filename, "wb") as f:
            f.write(audio)

        # Return an AIMessage containing a local file URL
        audio_url = f"file://{filename}"
        ai_msg = AIMessage(content=[{"type": "audio_url", "audio_url": {"url": audio_url}}])
        return ai_msg


class OpenAIAudioToText(BaseChatOpenAI):
    """Transcribe audio files to text using OpenAI Whisper API."""

    @property
    def _llm_type(self) -> str:
        return "openai-audio-to-text"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = []
        for message in messages:
            ai_msg = self._convert_audio_to_text(message, **kwargs)
            gen = ChatGeneration(message=ai_msg)
            result.append(gen)
        return ChatResult(generations=result)

    def _convert_audio_to_text(self, message: BaseMessage, **kwargs) -> AIMessage:
        # Extract the first assistant message containing audio URL
        audio_url = None
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "audio_url":
                audio_url = block.get("audio_url", {}).get("url")
                break
        if not audio_url:
            raise ValueError("Audio URL is required for transcription.")

        # Download audio bytes
        audio_bytes = urlopen(audio_url).read()

        # Call OpenAI Whisper transcription endpoint
        translation = False
        if "translation" in kwargs:
            kwargs.pop("translation")
            translation = bool(kwargs["translation"])
        if translation:
            resp = self.root_client.audio.translations.create(model=self.model_name, file=audio_bytes, **kwargs)
        else:
            resp = self.root_client.audio.transcriptions.create(model=self.model_name, file=audio_bytes, **kwargs)
        text = resp.text

        # Return the transcribed text as AIMessage
        ai_msg = AIMessage(content=[{"type": "text", "text": text}])
        return ai_msg
