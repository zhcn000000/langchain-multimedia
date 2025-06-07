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

### OpenAITextToAudio Example
```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia.audio import OpenAITextToAudio

model = OpenAITextToAudio(
    base_url="https://api.example.com",
    api_key="YOUR_API_KEY",
    model="voice-1",
)

message = HumanMessage(content=[{"type": "text", "text": "Hello, world"}])
ai_message = model.invoke(input=[message])

'''
ai_message.content like
[{"type": "audio_file", "path": "/path/to/generated_audio.mp3"}]
'''
```

### OpenAITextToImage Example
```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_multimedia.image import OpenAITextToImage

model = OpenAITextToImage(
    base_url="https://api.example.com",
    api_key="YOUR_API_KEY",
    model="vision-1",
)

message = HumanMessage(content=[{"type": "text", "text": "Generate a landscape photo with mountains and a river"}])
ai_message = model.invoke(input=[message])
'''
ai_message.content like
[{"type": "image_file", "path": "/path/to/generated_image.png"}]
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
