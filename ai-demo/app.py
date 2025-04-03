import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import numpy as np
import librosa
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import logging
from pydub import AudioSegment
from deepface import DeepFace  # For facial emotion detection
import cv2  # For capturing video frames
import base64
import PyPDF2   
import docx  # Use python-docx explicitly
import spacy
import random
from elevenlabs.client import ElevenLabs  # Import ElevenLabs client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Configure Gemini API
genai.configure(api_key="AIzaSyAU3lqb-xlubVzrTyDslPaxX_tmUD1i_eo")

# Configure ElevenLabs API
client = ElevenLabs(api_key="()sk_ca92f6843b635d499e0cc88faeb59178871e0b55babeba7c")

# List available models and select one
def get_available_model():
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                logger.info(f"Available model: {model.name}")
                if 'gemini-1.5-flash' in model.name:
                    logger.info(f"Using model: {model.name}")
                    return genai.GenerativeModel(model.name)
        logger.info("No preferred model found, using gemini-1.5-flash")
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        return genai.GenerativeModel('gemini-1.5-flash')

model = get_available_model()

# Initialize sentiment analyzer and spaCy
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Global variables for conversation, interview type and tech score
conversation_history = []
current_interview_type = None
extracted_skills = []  # Global to store resume skills

# For tech interview scoring
tech_question_count = 0
tech_score = 0
MAX_TECH_QUESTIONS = 10

# List of HR questions to cycle through
hr_questions = [
    "Tell me about a time you worked in a team. How did you contribute?",
    "Describe a challenging situation at work and how you handled it.",
    "What motivates you to perform well in your job?",
    "How do you handle stress and pressure in the workplace?",
    "Where do you see yourself in five years?",
    "Tell me about a time you failed. How did you deal with it?",
    "What is your greatest strength and how have you used it in a professional setting?",
    "How do you prioritize your tasks when you have multiple deadlines?"
]

# Counter to track the current HR question index
hr_question_index = 0

# Function to evaluate a tech response (liberal marking: answer must not be one word)
def gemini_mark_answer(answer_text):
    """
    Ask Gemini to mark the candidate's technical answer on a scale from 0.0 to 1.0.
    The prompt instructs it to return only a number (e.g., 0.75).
    In case of failure, falls back to the local evaluate_response.
    """
    try:
        prompt = (f"Mark the following technical answer on a scale from 0.0 to 1.0 for "
                  f"accuracy and relevance: '{answer_text}'. Return only the mark as a floating point number (for example, 0.75).")
        response_obj = model.generate_content(prompt)
        mark_str = response_obj.text.strip().split()[0]  # Expecting the first token is the mark
        mark = float(mark_str)
        return mark
    except Exception as e:
        logger.error(f"Error in marking answer with Gemini: {e}")
        # If Gemini fails, fall back to the local evaluator.
        return evaluate_response(answer_text)

# Function to detect facial emotion using DeepFace
def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result and isinstance(result, list) and len(result) > 0:
            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            return emotions, dominant_emotion
        else:
            logger.warning("No emotions detected in the frame")
            return None, None
    except Exception as e:
        logger.error(f"Error in emotion detection: {e}")
        return None, None

# Function to capture video frame from the frontend
def capture_frame(image_data):
    try:
        if "," in image_data:
            header, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
        else:
            try:
                binary_data = base64.b64decode(image_data)
            except Exception as e:
                logger.error(f"Failed to decode image data: {e}")
                return None
        
        temp_path = "temp_frame.jpg"
        with open(temp_path, "wb") as f:
            f.write(binary_data)
        
        frame = cv2.imread(temp_path)
        if frame is None:
            logger.error("Failed to read image from file")
            return None
        return frame
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None

# Generate technical question based on extracted skills and conversation history
def generate_tech_question(response=None):
    global extracted_skills, conversation_history, tech_question_count
    if not model:
        logger.warning("Model not initialized, using fallback question")
        return "What is the difference between a list and a tuple in Python?"
    try:
        context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history])
        if extracted_skills and len(extracted_skills) > 0:
            skill = random.choice(extracted_skills)
            if not response:
                prompt = (f"Given the conversation context:\n{context}\n"
                          f"Ask a basic technical interview question about {skill} that requires more than a one-word answer.")
            else:
                prompt = (f"Given the conversation context:\n{context}\n"
                          f"Based on the response: '{response}', ask a follow-up technical question about {skill} that builds on the previous answer.")
        else:
            if not response:
                prompt = (f"Given the conversation context:\n{context}\n"
                          "Ask a basic technical interview question about Python that requires more than a one-word answer.")
            else:
                prompt = (f"Given the conversation context:\n{context}\n"
                          "Based on the response: '{response}', ask a follow-up technical question about Python that builds on the previous answer. only the follow up question but nothning else dont show me your thinking")
        response_obj = model.generate_content(prompt)
        question = response_obj.text
        logger.info(f"Generated tech question: {question}")
        return question
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}")
        return "What is the difference between a list and a tuple in Python?"

# Generate HR question based on conversation history
def generate_hr_question():
    global hr_question_index, hr_questions, conversation_history
    context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history])
    try:
        if len(conversation_history) > 1:
            prompt = (f"Given the conversation context:\n{context}\n"
                      "Ask a follow-up HR interview question.")
            response_obj = model.generate_content(prompt)
            question = response_obj.text
        else:
            question = hr_questions[hr_question_index]
            hr_question_index = (hr_question_index + 1) % len(hr_questions)
        logger.info(f"Generated HR question: {question}")
        return question
    except Exception as e:
        logger.error(f"Error generating HR question: {e}")
        question = hr_questions[hr_question_index]
        hr_question_index = (hr_question_index + 1) % len(hr_questions)
        return question

# Convert text to speech using ElevenLabs
def text_to_speech(text, filename="question.mp3"):
    try:
        audio = client.generate(
            text=text,
            voice="Rachel",
            model="eleven_monolingual_v1",
            voice_settings={
                "stability": 0.7,
                "similarity_boost": 0.8
            }
        )
        audio_path = os.path.join("static", filename)
        with open(audio_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        logger.info(f"Audio file saved at: {audio_path}")
        return filename
    except Exception as e:
        logger.error(f"Error in text-to-speech with ElevenLabs: {e}")
        return None

# Analyze speech for pitch and energy
def analyze_speech(audio_file):
    try:
        y, sr = librosa.load(audio_file)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        energy = np.mean(librosa.feature.rms(y=y))
        return pitch_mean, energy
    except Exception as e:
        logger.error(f"Error in speech analysis: {e}")
        return 0, 0

# Analyze soft skills
def analyze_soft_skills(text, pitch, energy, emotions=None):
    try:
        sentiment = analyzer.polarity_scores(text)
        confidence = "High" if pitch > 100 else "Low"
        enthusiasm = "High" if energy > 0.1 else "Low"
        positivity = sentiment['compound']
        emotion_feedback = ""
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            emotion_feedback = f"Dominant Emotion: {dominant_emotion}"
        return {
            "confidence": confidence,
            "enthusiasm": enthusiasm,
            "positivity": positivity,
            "emotion_feedback": emotion_feedback,
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Error in soft skills analysis: {e}")
        return {"confidence": "Unknown", "enthusiasm": "Unknown", "positivity": 0.0, "emotion_feedback": "Unknown"}

# Convert audio to WAV format
def convert_to_wav(input_file, output_file="response.wav"):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        logger.info(f"Converted audio to WAV: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {e}")
        return None

# Extract text from resume
def extract_text(file):
    try:
        text = ""
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file)
            for paragraph in doc.paragraphs:
                para_text = paragraph.text or ""
                text += para_text + "\n"
        else:
            logger.error("Unsupported file format")
            return ""
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

# Extract skills from text
def extract_skills(text):
    global extracted_skills
    try:
        skills_list = [
            "python", "java", "c", "c++", "javascript", "sql", "html", "css",
            "ruby", "php", "go", "rust", "typescript", "kotlin", "swift",
            "scala", "r", "perl", "matlab", "bash", "powershell",
            "flask", "django", "spring", "react", "angular", "vue.js", "node.js",
            "express", "laravel", "rails", "aspnet", "svelte",
            "android", "ios", "flutter", "xamarin", "react native",
            "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
            "opencv", "theano", "caffe", "mxnet",
            "hadoop", "spark", "kafka", "flink", "airflow", "tableau", "power bi",
            "dask", "apache hive", "apache pig",
            "mongodb", "postgresql", "mysql", "oracle", "sqlite", "cassandra",
            "redis", "elasticsearch", "mariadb", "firebase",
            "aws", "azure", "google cloud", "ibm cloud", "oracle cloud", "heroku",
            "digitalocean", "linode",
            "docker", "kubernetes", "jenkins", "ansible", "terraform", "chef",
            "puppet", "circleci", "travis ci", "github actions", "gitlab ci",
            "bitbucket pipelines",
            "git", "svn", "mercurial", "perforce",
            "apache", "nginx", "tomcat", "iis", "haproxy", "traefik", "dns",
            "dhcp", "iptables", "wireguard",
            "selenium", "junit", "pytest", "mocha", "jest", "cypress", "postman",
            "soapui",
            "linux", "windows server", "macos", "Ubuntu", "centos", "redhat",
            "vim", "emacs", "vscode", "intellij", "eclipse", "grafana",
            "prometheus", "loki", "jaeger", "rabbitmq", "celery", "gunicorn",
            "supervisor", "logstash", "kibana", "splunk",
            "metasploit", "nmap", "wireshark", "burp suite", "owasp zap",
            "nessus", "qualys",
            "arduino", "raspberry pi", "esp32", "stm32", "zigbee", "mqtt",
            "graphql", "rest", "soap", "websocket", "grpc", "protobuf",
            "webpack", "babel", "eslint", "prettier", "rollup"
        ]
        doc = nlp(text.lower())
        extracted_skills = set()
        for i in range(len(doc)):
            for skill in skills_list:
                skill_tokens = skill.split()
                if len(skill_tokens) == 1:
                    if doc[i].text == skill:
                        extracted_skills.add(skill)
                else:
                    window = doc[i:i + len(skill_tokens)]
                    if all(t.text == skill_tokens[j] for j, t in enumerate(window)) and len(window) == len(skill_tokens):
                        extracted_skills.add(skill)
        extracted_skills = list(extracted_skills)
        logger.info(f"Extracted skills: {extracted_skills} (Count: {len(extracted_skills)})")
        return extracted_skills
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global conversation_history, current_interview_type, tech_question_count, tech_score, hr_question_index
    try:
        interview_type = request.json.get('type')
        if not interview_type:
            logger.error("Interview type not provided")
            return jsonify({"error": "Interview type not provided"}), 400

        conversation_history = []
        current_interview_type = interview_type
        hr_question_index = 0
        if interview_type == "tech":
            tech_question_count = 0
            tech_score = 0
            question = generate_tech_question()
        else:
            question = generate_hr_question()
        
        audio_file = text_to_speech(question)
        if not audio_file:
            return jsonify({"error": "Failed to generate audio"}), 500

        conversation_history.append({"role": "interviewer", "text": question})
        logger.info(f"Interview started, type: {interview_type}, question: {question}")
        return jsonify({"question": question, "audio": audio_file})
    except Exception as e:
        logger.error(f"Error in start_interview: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/submit_response', methods=['POST'])
def submit_response():
    global conversation_history, current_interview_type, extracted_skills, tech_question_count, tech_score
    try:
        interview_type = request.form.get('type') or current_interview_type
        if not interview_type:
            logger.error("Interview type not provided and current_interview_type is None")
            return jsonify({"error": "Interview type not provided"}), 400

        logger.info(f"Processing response for interview type: {interview_type}")

        if 'audio' not in request.files:
            logger.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        temp_audio_path = os.path.join("temp_audio")
        audio_file.save(temp_audio_path)
        audio_path = convert_to_wav(temp_audio_path)
        if not audio_path:
            return jsonify({"error": "Failed to convert audio to WAV"}), 500
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        logger.info(f"Audio file saved at: {audio_path}")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            response_text = recognizer.recognize_google(audio)
            logger.info(f"Recognized text: {response_text}")
        except sr.UnknownValueError:
            response_text = "Could not understand audio."
            logger.warning("Speech recognition failed: Could not understand audio")
        
        conversation_history.append({"role": "user", "text": response_text})
        
        # Evaluate tech answer if applicable
        if interview_type == "tech":
            mark = gemini_mark_answer(response_text)
            tech_score += mark
            tech_question_count += 1
        
        image_data = request.form.get('image_data')
        if image_data:
            frame = capture_frame(image_data)
            if frame is not None:
                try:
                    emotions, dominant_emotion = detect_emotion(frame)
                    if emotions is not None:
                        logger.info(f"Detected emotions: {emotions}, Dominant Emotion: {dominant_emotion}")
                except Exception as e:
                    logger.error(f"Error processing facial emotions: {e}")
        
        # If tech interview reached MAX_TECH_QUESTIONS, finish interview
        if interview_type == "tech" and tech_question_count >= MAX_TECH_QUESTIONS:
            final_message = f"Tech Interview Completed. Your score is {tech_score} out of {MAX_TECH_QUESTIONS}."
            audio_file = text_to_speech(final_message)
            conversation_history.append({"role": "interviewer", "text": final_message})
            logger.info(f"Tech interview completed. Score: {tech_score}/{MAX_TECH_QUESTIONS}")
            return jsonify({"question": final_message, "audio": audio_file})
        else:
            if interview_type == "tech":
                next_question = generate_tech_question(response_text)
            else:
                next_question = generate_hr_question()

        audio_file = text_to_speech(next_question)
        if not audio_file:
            return jsonify({"error": "Failed to generate audio"}), 500

        conversation_history.append({"role": "interviewer", "text": next_question})
        logger.info(f"{interview_type} interview, next question: {next_question}")
        return jsonify({"question": next_question, "audio": audio_file})
    except Exception as e:
        logger.error(f"Error in submit_response: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    global extracted_skills
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
        if file:
            text = extract_text(file)
            if not text:
                return jsonify({"error": "Failed to extract text from resume"}), 500
            skills = extract_skills(text)
            extracted_skills = skills
            logger.info(f"Extracted skills: {extracted_skills}")
            return jsonify({"skills": extracted_skills})
    except Exception as e:
        logger.error(f"Error in upload_resume: {e}")
        return jsonify({"error": str(e)}), 400

# New route to serve candidate profile for tech score
@app.route('/profile', methods=['GET'])
def profile():
    global tech_score, MAX_TECH_QUESTIONS
    return jsonify({"score": tech_score, "max_score": MAX_TECH_QUESTIONS})

if __name__ == '__main__':
    app.run(debug=True)