import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional
from urllib.request import urlopen

import magic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class OpenAITextToImage(BaseChatOpenAI):
    """
    Uses OpenAI Image API to generate images from text prompts.
    """

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
        input_image_url = None
        if not isinstance(message, str):
            for block in message.content:
                if not isinstance(block, str) and block["type"] == "image_url":
                    input_image_url = block["image_url"]["url"]
                    break
        if input_image_url is None:
            # Call OpenAI Image API to generate an image
            response = self.root_client.images.generate(
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
        elif prompt is None or prompt.strip() == "":
            # If no prompt is provided, use the input image URL for image generation
            input_image = urlopen(input_image_url).read()
            response = self.root_client.images.create_variation(
                model=self.model_name,
                image=input_image,
                **kwargs,
            )
        else:
            # If an input image URL is provided, use it for image generation
            input_image = urlopen(input_image_url).read()
            response = self.root_client.images.edit(
                model=self.model_name,
                prompt=prompt,
                image=input_image,
                **kwargs,
            )
        output_image_url = response.data[0].url
        image = urlopen(output_image_url).read()
        cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
        cache_dir.mkdir(parents=True, exist_ok=True)
        mime = magic.from_buffer(image, mime=True)
        ext = mime.split("/")[-1]
        filename = cache_dir / f"{uuid.uuid4()}.{ext}"
        with open(filename, "wb") as f:
            f.write(image)

        # Retrieve the URL of the first generated image
        image_url = f"file://{filename}"

        # Wrap the result into an AIMessage and ChatResult
        ai_msg = AIMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            ]
        )
        return ai_msg
