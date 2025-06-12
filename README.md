# LangChain Multimedia

## Overview
This project leverages [LangChain](https://github.com/langchain-ai/langchain) to process and generate multimedia content (audio, video, images) with plugin-based extensions and custom model integration.

## Features
- Text-to-Audio/Video/Image
- OCR  
- Audio-to-Text
- Configurable Parameters Management  

## Support Model Providers
- Xinference
- OpenAI
- Stable Diffusion

## Installation
1. Clone the repository  
   ```bash
   git clone https://github.com/zhcn000000/langchain-multimedia.git
   cd langchain-multimedia
   ```
2. Create and activate a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Edit `config.yaml` to set your API key, model parameters, etc.  
2. Run an example script:  
   ```bash
   python examples/audio_to_text.py
   ```

### OpenAIAudioGenerator Example
```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia import OpenAIAudioGenerator

model = OpenAIAudioGenerator(
    base_url="https://api.example.com",
    api_key="YOUR_API_KEY",
    model="voice-1",
)
model.voice = "en-US-Wavenet-D"  # Set the voice model
prompt = "Hello, world"
response = model.invoke(input=prompt)

'''
response = "/path/to/generated_audio.mp3"
'''
```

### OpenAIImageGenerator Example
```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia import OpenAIImageGenerator

model = OpenAIImageGenerator(
    base_url="https://api.example.com",
    api_key="YOUR_API_KEY",
    model="vision-1",
)

prompt = "Generate a landscape photo with mountains and a river"
response = model.invoke(input=prompt)
'''
response = "/path/to/generated_image.png"
'''
```

## OpenAITranscriber Example
```python
from langchain_multimedia import OpenAITranscriber
from pathlib import Path
audio_file = "/path/to/audio.mp3"
audio_data = Path(audio_file).read_bytes()

model = OpenAITranscriber(
    base_url="https://api.example.com",
    api_key="YOUR_API_KEY",
    model="whisper-1",
)

response = model.invoke(input=audio_data)
'''
response = "Transcribed text from the audio file"
'''
```

## Configuration
In `tests/api.json`, you can configure:
- `api_key`: API key for model service  
- `model_name`: Selected model name  
- `timeout`: Request timeout in seconds  
- Parameters for plugins and extensions  

## Tested Models

- Currently only OpenAI and XInference image and audio models have been tested; other models are not yet tested.

## Project Structure
```
.
├── examples/               Example scripts
├── langchain_multimedia/   Core modules
├── tests/                  Unit tests
├── tests/api.json         tests api config file
├── requirements.txt        Dependencies
└── README.md               Project documentation
```

## License
This project is licensed under the MIT License. 
