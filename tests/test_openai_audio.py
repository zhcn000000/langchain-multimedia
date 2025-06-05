from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia.audio import OpenAITextToAudio,OpenAIAudioToText,XinferenceAudioToText
from urllib.request import urlopen
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "api.json")
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)

XINFERENCE_HOST = "localhost"
XINFERENCE_PORT = 44000

def test_generate_audio():
    model = OpenAITextToAudio(
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
    out_audio_url = ai_message.content[0]["audio_url"]["url"]
    audio_data = urlopen(out_audio_url).read()
    with open("test.mp3", "wb") as f:
        f.write(audio_data)
    model = OpenAIAudioToText(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["sound-model"],
    )
    message = HumanMessage(
        content=[
            {"type": "audio_url", "audio_url": {"url": out_audio_url}},
        ]
    )
    ai_message = model.invoke(input=[message])
    content = ai_message.text()
    print(content)
    assert isinstance(content,str)
    assert isinstance(ai_message, AIMessage)
