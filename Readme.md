# 🧠 AI Interviewer App

**AI Interviewer** is an intelligent HR & technical interview simulator that dynamically generates questions using Gemini, analyzes facial expressions using DeepFace, and evaluates answers using VADER sentiment analysis. The app also integrates ElevenLabs for text-to-speech and captures audio responses in real-time.

---

## 🚀 Features

- 🎯 **Dynamic Question Generation**
  - Uses **Gemini API** to parse skills from the uploaded resume (PDF or text)
  - Generates relevant, role-specific interview questions on the fly

- 🗣️ **Voice Interaction**
  - Converts Gemini's questions to speech using **ElevenLabs API**
  - Records and transcribes candidate's spoken answers

- 🧠 **HR Analysis**
  - Uses **VADER Sentiment Analysis** to detect confidence and tone in answers
  - **DeepFace** analyzes facial expressions for emotion recognition

- 📊 **Feedback Report**
  - Generates a summary report for candidates with:
    - Score
    - Strengths & weaknesses
    - Suggested improvements

---

## 🛠️ Tech Stack

- **Backend:** Python + Flask
- **AI Models/APIs:** Gemini, DeepFace, VADER, ElevenLabs
- **Audio Handling:** PyDub, SpeechRecognition
- **Frontend/UI:** Streamlit (optional for testing)

---

## 📦 Installation

```bash
git clone https://github.com/Chad-007/mini-project.git
cd ai-demo
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

> Configure your `.env` file with API keys for Gemini, ElevenLabs, etc.

---

## 📄 .env Example

```
GEMINI_API_KEY=your_gemini_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

---

## 🧪 Sample Flow

1. Upload resume
2. Gemini extracts skills & generates questions
3. Questions converted to voice via ElevenLabs
4. User answers verbally
5. DeepFace & VADER evaluate emotion & sentiment
6. Gemini provides feedback and next question

---

## 📬 Contact

For issues or contributions, open a GitHub issue or reach out at: alansebastian484@gmail.com

---

## 📄 License

MIT License © 2025 Chad-007