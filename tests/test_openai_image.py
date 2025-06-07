from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia.image import OpenAITextToImage
from urllib.request import urlopen
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "api.json")
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)

XINFERENCE_HOST = "localhost"
XINFERENCE_PORT = 44000

def test_generate_audio_creates_file_and_returns_ai_message():
    model = OpenAITextToImage(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["image-model"],
    )
    text = "A beautiful sunset over the mountains"

    message = HumanMessage(
        content=[
            {"type": "text", "text": text},
        ]
    )
    ai_message = model.invoke(input=[message])
    image_file = Path(ai_message.content[0]["path"])
    image_data = image_file.read_bytes()
    with open("test.png", "wb") as f:
        f.write(image_data)
    assert isinstance(ai_message, AIMessage)
