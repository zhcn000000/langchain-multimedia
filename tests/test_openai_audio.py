from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from openai.types.beta.threads import image_file

from langchain_multimedia import OpenAIAudioGenerator,OpenAITranscriber
from urllib.request import urlopen
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "api.json")
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)

def test_generate_audio():
    model = OpenAIAudioGenerator(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["voice-model"],
    )

    text = "Hallo World"
    # 构造消息
    model.voice = "FunAudioLLM/CosyVoice2-0.5B:alex"
    result = model.invoke(input=text)
    assert isinstance(result, str)
    audio_data = Path(result).read_bytes()
    with open("test.mp3", "wb") as f:
        f.write(audio_data)

    model = OpenAITranscriber(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["sound-model"],
    )

    result = model.invoke(input=result)
    print(result)
    assert isinstance(result,str)
