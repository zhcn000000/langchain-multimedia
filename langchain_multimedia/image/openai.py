from typing import Any, Dict, List, Optional
from urllib.request import urlopen

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_multimedia.core import BaseGenerationOpenAI, GenerationType
from langchain_multimedia.utils.helpers import _build_image, _find_image


class OpenAIImageGenerator(BaseGenerationOpenAI):
    """
    Uses OpenAI Image API to generate images from text prompts.
    """
    @property
    def _generator_type(self):
        return GenerationType.ImageGenerator

    @property
    def _llm_type(self) -> str:
        # Identifier for this LLM type
        return "openai-text-to-image"

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
            ai_msg = self._convert_text_to_image(message)
            gen = ChatGeneration(message=ai_msg)
            results.append(gen)
        return ChatResult(generations=results)

    def _convert_text_to_image(self, message: BaseMessage, **kwargs: Any) -> AIMessage:
        # Extract prompt from the first message
        prompt = message.text()
        input_image = _find_image(message)
        if input_image is None:
            # Call OpenAI Image API to generate an image
            response = self.root_client.images.generate(
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
        elif prompt is None or prompt.strip() == "":
            response = self.root_client.images.create_variation(
                model=self.model_name,
                image=input_image,
                **kwargs,
            )
        else:
            response = self.root_client.images.edit(
                model=self.model_name,
                prompt=prompt,
                image=input_image,
                **kwargs,
            )
        output_image_url = response.data[0].url
        output_image = urlopen(output_image_url).read()

        # Wrap the result into an AIMessage and ChatResult
        ai_msg = AIMessage(
            content=[_build_image(output_image)],
        )
        return ai_msg
