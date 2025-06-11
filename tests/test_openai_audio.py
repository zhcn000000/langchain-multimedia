from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from openai.types.beta.threads import image_file

from langchain_multimedia.audio import OpenAIAudioGenerator,OpenAITranscriptor
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

    message = HumanMessage(
        content=[
            {"type": "text", "text": text},
        ]
    )
    ai_message = model.invoke(input=[message], voice="FunAudioLLM/CosyVoice2-0.5B:alex",response_format="wav")
    assert isinstance(ai_message, AIMessage)
    audio_file = Path(ai_message.content[0]["path"])
    audio_data = audio_file.read_bytes()
    with open("test.mp3", "wb") as f:
        f.write(audio_data)
    model = OpenAITranscriptor(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["sound-model"],
    )
    message = ai_message
    ai_message = model.invoke(input=[message])
    content = ai_message.text()
    print(content)
    assert isinstance(content,str)
    assert isinstance(ai_message, AIMessage)
