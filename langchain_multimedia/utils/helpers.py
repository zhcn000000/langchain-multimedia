import tempfile
from mimetypes import guess_extension
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

from langchain_core.messages import BaseMessage
from magic import from_buffer

cache_dir = Path(tempfile.gettempdir()) / "langchain_multimedia"
cache_dir.mkdir(parents=True, exist_ok=True)


def _find_image(message: BaseMessage):
    image = None
    if not isinstance(message, str):
        for block in message.content:
            if isinstance(block, str):
                continue
            elif block.get("type") == "image_url":
                input_image_url = block.get("image_url", {}).get("url")
                image = urlopen(input_image_url).read()
            elif block.get("type") == "image":
                image = block.get("data")
            elif block.get("type") == "image_file":
                image_path = Path(block.get("path", ""))
                image = image_path.read_bytes()
    return image


def _find_audio(message: BaseMessage):
    audio = None
    if not isinstance(message, str):
        for block in message.content:
            if isinstance(block, str):
                continue
            elif block.get("type") == "audio_url":
                input_audio_url = block.get("audio_url", {}).get("url")
                audio = urlopen(input_audio_url).read()
            elif block.get("type") == "audio":
                audio = block.get("data")
            elif block.get("type") == "audio_file":
                audio_path = Path(block.get("path", ""))
                audio = audio_path.read_bytes()
    return audio


def _build_image(output_image):
    mime = from_buffer(output_image, mime=True)
    ext = guess_extension(mime)
    filename = cache_dir / f"{uuid4()}{ext}"
    filename.write_bytes(output_image)
    return {
        "type": "image_file",
        "path": str(filename),
    }


def _build_audio(output_audio):
    mime = from_buffer(output_audio, mime=True)
    ext = guess_extension(mime)
    filename = cache_dir / f"{uuid4()}{ext}"
    filename.write_bytes(output_audio)
    return {
        "type": "audio_file",
        "path": str(filename),
    }


def _build_video(output_video):
    mime = from_buffer(output_video, mime=True)
    ext = guess_extension(mime)
    filename = cache_dir / f"{uuid4()}{ext}"
    filename.write_bytes(output_video)
    return {
        "type": "video_file",
        "path": str(filename),
    }
