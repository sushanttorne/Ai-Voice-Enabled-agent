
## ✅ `README.md` 

## 🎧 Azure Voice AI Assistant – Multi-Agent Web Interface

A sleek, responsive **AI voice agent** web app built with:
- ✅ Microsoft Project AI Agents via `azure-ai-projects`
- 🗣 Azure Cognitive Services for real-time **Speech-to-Text** (STT) and **Text-to-Speech** (TTS)
- ⚡ Dynamic frontend UI (HTML5 + JS) with VAD (Voice Activity Detection), agent chat, and performance metrics

![UI Preview](./90a334a0-f267-4bc5-92cd-1754ef41404b.png)

---

## 🔧 Key Features

- 🎯 **Plug-and-Play Azure Agent ID**: Easily switch agents by changing `AGENT_ID` in `back1.py`
- 🎙️ Voice-first experience: Real-time voice input, agent processing, and spoken responses
- 🧠 Fallback "mock mode" when Azure is not configured (for local/dev use)
- 📊 Live metrics: Response time, voice level, TTS status, speech recognition
- 💬 Rich conversation canvas UI

## 🚀 Setup Instructions
### 1. Clone the Repository

```bash
git clone https://github.com/your-username/azure-voice-agent.git
cd azure-voice-agent
````

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Update Azure Credentials

Create a `.env` file or modify directly in `back1.py`:

```env
AZURE_CONNECTION_STRING=your_connection_string
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=eastus2
AGENT_ID=your_agent_id
```

### 5. Run the Backend

```bash
python back1.py
```

### 6. Open the Frontend

Open `app1.html` directly in your browser, or serve via a local HTTP server:

```bash
python -m http.server 8080
```

> Default backend URL: `http://127.0.0.1:5000`

---

## 🗃 File Structure

```
.
├── app1.html             # Frontend with full voice UI
├── back1.py              # Flask API with Azure AI & Speech SDK
├── requirements.txt      # Python dependencies
├── .env.example          # Example config for Azure
├── 90a334a0...png        # Screenshot (for README)
└── README.md
```

---

## 📦 Dependencies

```txt
Flask==2.3.3
flask-cors==4.0.0
azure-ai-projects==1.0.0b10
azure-identity==1.15.0
azure-cognitiveservices-speech==1.34.1
```

---

## 📌 Notes

* **Supports Azure AI Foundry agents** out-of-the-box
* **Switch agents** easily by changing just the `AGENT_ID`
* Automatically falls back to mock mode if Azure credentials are missing
* Runs locally with no build tools

---

## 📄 License

MIT License

````

---

## ✅ `.env.example`

```env
AZURE_CONNECTION_STRING=eastus2.api.azureml.ms;...;resource-group;project-name
AZURE_SPEECH_KEY=your_speech_key_here
AZURE_SPEECH_REGION=eastus2
AGENT_ID=your_agent_id_here
````

---

## ✅ Shell Script to Run Locally (optional)

```bash
#!/bin/bash
source venv/bin/activate
export FLASK_APP=back1.py
python back1.py
```

Name it `run.sh` and make it executable with `chmod +x run.sh`.

---
