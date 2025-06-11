from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI

from langchain_multimedia.utils.helpers import _build_audio, _find_audio
from langchain_multimedia.core import BaseGenerationOpenAI,GenerationType

class OpenAIAudioGenerator(BaseGenerationOpenAI):
    """Generate audio from text prompts using OpenAI TTS API."""

    @property
    def _llm_type(self) -> str:
        return "openai-text-to-audio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _generator_type(self):
        return GenerationType.AudioGenerator

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

        return AIMessage(content=[_build_audio(audio)])


class OpenAITranscriptor(BaseChatOpenAI):
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
        input_audio = _find_audio(message)
        # Call OpenAI Whisper transcription endpoint
        translation = False
        if "translation" in kwargs:
            kwargs.pop("translation")
            translation = bool(kwargs["translation"])
        if translation:
            resp = self.root_client.audio.translations.create(model=self.model_name, file=input_audio, **kwargs)
        else:
            resp = self.root_client.audio.transcriptions.create(model=self.model_name, file=input_audio, **kwargs)
        text = resp.text
        # Return the transcribed text as AIMessage
        ai_msg = AIMessage(content=[{"type": "text", "text": text}])
        return ai_msg
