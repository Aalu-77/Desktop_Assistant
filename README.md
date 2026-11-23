# ULTRON – Personal AI Desktop Assistant

ULTRON is a local AI-powered desktop assistant inspired by JARVIS. It combines voice interaction, natural-language conversations, and computer monitoring with locally running AI models.

## Features

- Voice activation using the wake phrase **“Hi Ultron”** and push-to-talk
- Natural conversations powered by Ollama
- Local speech recognition using Whisper (ONNX)
- Spoken responses using pyttsx3
- CPU, RAM, and GPU monitoring
- News headlines and summaries
- Today's schedule (currently using sample events until a calendar provider is connected)

## Tech Stack

- Python
- Ollama (`llama3.2:3b`)
- Whisper / ONNX
- FastAPI
- PySide6
- pyttsx3
- NumPy and psutil

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/)
- Microphone and speakers
- NVIDIA GPU is optional

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Aalu-77/Desktop_Assistant.git
   cd ULTRON
   ```

2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the Ollama model:

   ```bash
   ollama pull llama3.2:3b
   ```

4. Make sure Ollama is running, then start the backend:

   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8765
   ```

5. Launch the desktop interface using the project's launcher.

## Project Status

ULTRON is an ongoing personal project. Features and architecture may change as development continues.

## Planned Improvements

- Connect a real calendar provider
- Persistent conversation memory
- Application launching and desktop controls
- Additional assistant tools and integrations

---

*Built with Python and Ollama, inspired by JARVIS.*
