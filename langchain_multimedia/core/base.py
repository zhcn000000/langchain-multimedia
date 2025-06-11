from abc import abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from enum import Enum

from langchain_openai.chat_models.base import BaseChatOpenAI


class GenerationType(Enum):
    """Enum for different model types."""
    ImageGenerator = "ImageGenerator"
    AudioGenerator = "AudioGenerator"
    VideoGenerator = "VideoGenerator"


class GenerationTool(BaseTool):
    """Base class for multimedia generation tools."""
    generator = None

    def __init__(self,generator,generation_type: GenerationType, **kwargs):
        """Initialize the generation tool with a specific type."""
        super().__init__(**kwargs)
        if generation_type == GenerationType.ImageGenerator:
            self.name = "ImageGenerator"
            self.description = "Generates images from text prompts."
        elif generation_type == GenerationType.AudioGenerator:
            self.name = "AudioGenerator"
            self.description = "Generates audio from text prompts."
        elif generation_type == GenerationType.VideoGenerator:
            self.name = "VideoGenerator"
            self.description = "Generates videos from text prompts."
        else:
            raise ValueError(f"Unsupported generation type: {generation_type}")

    def _run(self,prompt: str) -> str:
        """Run the image generation tool."""
        message = HumanMessage(
            content=prompt
        )
        response = self.generator.invoke([message])
        path = response.generations[0].message.content[0].path
        return path

    async def _arun(self,prompt: str) -> str:
        message = HumanMessage(
            content=prompt
        )
        response =await self.generator.ainvoke([message])
        path = response.generations[0].message.content[0].path
        return path



class BaseGenerationModel(BaseChatModel):
    """Base class for generation models that extends BaseChatModel."""


    """Delete some attributes that are not applicable to multimedia models."""
    stream=None
    astream=None
    bind_tools=None
    with_struct_output=None

    @abstractmethod
    def _generator_type(self) -> GenerationType:
        """Return the type of the generator."""
        raise NotImplementedError("Subclasses must implement this method.")

    def as_atool(self) -> GenerationTool:
        generation_type = self._generator_type()
        return GenerationTool(
            generator=self,
            generation_type=generation_type,
            name=self.name,
            description=self.description
        )

class BaseGenerationOpenAI(BaseChatOpenAI):
    """Base class for generation models that extends BaseChatOpenAI."""


    """Delete some attributes that are not applicable to multimedia models."""
    stream=None
    astream=None
    bind_tools=None
    with_struct_output=None

    @abstractmethod
    def _generator_type(self) -> GenerationType:
        """Return the type of the generator."""
        raise NotImplementedError("Subclasses must implement this method.")

    def as_atool(self) -> GenerationTool:
        generation_type = self._generator_type()
        return GenerationTool(
            generator=self,
            generation_type=generation_type,
            name=self.name,
            description=self.description
        )