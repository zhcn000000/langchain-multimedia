from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia import OpenAIImageGenerator
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "api.json")
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)


def test_generate_audio_creates_file_and_returns_ai_message():
    model = OpenAIImageGenerator(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["image-model"],
    )
    text = "A beautiful sunset over the mountains"
    # 构造消息
    response = model.invoke(input=text)
    image_file = Path(response)
    image_data = image_file.read_bytes()
    with open("test.png", "wb") as f:
        f.write(image_data)
    assert isinstance(response, str)
