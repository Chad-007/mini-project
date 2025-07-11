import os
import uuid
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import numpy as np
import librosa
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from pydub import AudioSegment
from deepface import DeepFace
import cv2
import base64
import PyPDF2
import docx
import spacy
import random
from elevenlabs.client import ElevenLabs
from supabase import create_client, Client
from datetime import datetime
import json
from dotenv import load_dotenv
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABSE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')

GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_KEY")
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

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

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

conversation_history = []
current_interview_type = None
extracted_skills = []
current_user_id = None

tech_question_count = 0
tech_score = 0
MAX_TECH_QUESTIONS = 10

hr_question_count = 0
hr_score = 0
MAX_HR_QUESTIONS = 8
hr_emotions_history = []
hr_soft_skills_history = []

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

hr_question_index = 0

def gemini_mark_hr_answer(answer_text):
    try:
        prompt = (f"Mark the following HR interview answer on a scale from 0.0 to 1.0 for be liberal in marking dont be very strict give them enough marks  "
                  f"clarity, relevance, and professionalism: '{answer_text}'. Return only the mark as a floating point number (e.g., 0.75).")
        response_obj = model.generate_content(prompt)
        mark = float(response_obj.text.strip().split()[0])
        return mark
    except Exception as e:
        logger.error(f"Error in marking HR answer: {e}")
        return 0.5

def gemini_mark_answer(answer_text):
    try:
        prompt = (f"Mark the following technical answer on a scale from 0.0 to 1.0 for give marks in a liberal fashion dont be very strict in giving marks give them enough marks"
                  f"accuracy and relevance: '{answer_text}'. Return only the mark as a floating point number (for example, 0.75).")
        response_obj = model.generate_content(prompt)
        mark = float(response_obj.text.strip().split()[0])
        return mark
    except Exception as e:
        logger.error(f"Error in marking answer with Gemini: {e}")
        return 0.5

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result and isinstance(result, list) and len(result) > 0:
            emotions = {k: float(v) for k, v in result[0]['emotion'].items()}
            dominant_emotion = result[0]['dominant_emotion']
            logger.info(f"Detected emotions: {emotions}, Dominant: {dominant_emotion}")
            return emotions, dominant_emotion
        else:
            logger.warning("No emotions detected in frame")
            return {}, None
    except Exception as e:
        logger.error(f"Error in emotion detection: {e}")
        return {}, None

def capture_frame(image_data):
    try:
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
        else:
            binary_data = base64.b64decode(image_data)
        temp_path = "temp_frame.jpg"
        with open(temp_path, "wb") as f:
            f.write(binary_data)
        frame = cv2.imread(temp_path)
        if frame is None:
            logger.error("Failed to load frame from temp file")
            return None
        return frame
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None

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
                          f"Ask a basic technical interview question about {skill} that requires more than a one-word answer. but dont ask questions that are too big aont ask any question that reauires me to submit the code snippet etc..")
            else:
                prompt = (f"Given the conversation context:\n{context}\n"
                          f"Based on the response: '{response}', ask a follow-up technical question about {skill} that builds on the previous answer. but not too big")
        else:
            if not response:
                prompt = (f"Given the conversation context:\n{context}\n"
                          "Ask a basic technical interview question about Python that requires more than a one-word answer.")
            else:
                prompt = (f"Given the conversation context:\n{context}\n"
                          "Based on the response: '{response}', ask a follow-up technical question about Python that builds on the previous answer.")
        response_obj = model.generate_content(prompt)
        question = response_obj.text.strip()
        logger.info(f"Generated tech question: {question}")
        return question
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}")
        return "What is the difference between a list and a tuple in Python?"

def generate_hr_question():
    global hr_question_index, hr_questions, conversation_history
    context = "\n".join([f"{entry['role']}: {entry['text']}" for entry in conversation_history])
    try:
        if len(conversation_history) > 1:
            prompt = (f"Given the conversation context:\n{context}\n"
                      "Ask a follow-up HR interview question that builds on the previous response. Return only the question.")
            response_obj = model.generate_content(prompt)
            question = response_obj.text.strip()
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


def text_to_speech(text, filename="question.mp3"):
    try:
        audio = client.generate(
            text=text,
            voice="Rachel",
            model="eleven_monolingual_v1",
            voice_settings={"stability": 0.7, "similarity_boost": 0.8}
        )
        audio_path = os.path.join("static", filename)
        with open(audio_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        logger.info(f"Audio file saved at: {audio_path}")
        return filename
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return None

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

def analyze_soft_skills(text, pitch, energy, emotions=None):
    try:
        sentiment = analyzer.polarity_scores(text)
        confidence = "High" if pitch > 100 else "Low"
        enthusiasm = "High" if energy > 0.1 else "Low"
        positivity = sentiment['compound']
        emotion_feedback = f"Dominant Emotion: {max(emotions, key=emotions.get)}" if emotions else "No emotions detected"
        return {
            "confidence": confidence,
            "enthusiasm": enthusiasm,
            "positivity": positivity,
            "emotion_feedback": emotion_feedback,
            "emotions": emotions or {}
        }
    except Exception as e:
        logger.error(f"Error in soft skills analysis: {e}")
        return {"confidence": "Unknown", "enthusiasm": "Unknown", "positivity": 0.0, "emotion_feedback": "Unknown", "emotions": {}}

def convert_to_wav(input_file, output_file="response.wav"):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        logger.info(f"Audio converted to WAV: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None

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
            "aws", "azure", "google cloud", "ibm cloud", "oraclevier cloud", "heroku",
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

@app.route('/sign_up', methods=['POST'])
def sign_up_route():
    global current_user_id
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        user_id = str(uuid.uuid4())
        existing_user = supabase.table("users").select("email").eq("email", email).execute()
        if existing_user.data:
            return jsonify({"error": "Email already exists"}), 400

        user_data = {"user_id": user_id, "email": email, "created_at": datetime.utcnow().isoformat()}
        user_response = supabase.table("users").insert(user_data).execute()
        if not user_response.data:
            return jsonify({"error": "Failed to insert user into database"}), 500

        profile_data = {
            "user_id": user_id,
            "tech_score": 0.0,
            "tech_max_score": 10,
            "hr_score": 0.0,
            "hr_max_score": 8,
            "hr_emotions": [],  
            "hr_soft_skills": [],  
            "last_updated": datetime.utcnow().isoformat()
        }
        profile_response = supabase.table("profiles").insert(profile_data).execute()
        if not profile_response.data:
            return jsonify({"error": "Failed to create user profile"}), 500

        current_user_id = user_id
        logger.info(f"User signed up with user_id: {user_id}")
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        logger.error(f"Error in sign-up route: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/sign_in', methods=['POST'])
def sign_in_route():
    global current_user_id
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        user = supabase.table("users").select("user_id").eq("email", email).execute()
        if not user.data:
            return jsonify({"error": "Invalid email or password"}), 400

        user_id = user.data[0]["user_id"]
        current_user_id = user_id
        logger.info(f"User signed in with user_id: {user_id}")
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        logger.error(f"Error in sign-in route: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global conversation_history, current_interview_type, tech_question_count, tech_score, hr_question_count, hr_score, hr_emotions_history, hr_soft_skills_history
    try:
        if not current_user_id:
            return jsonify({"error": "User not authenticated"}), 401

        interview_type = request.json.get('type')
        if not interview_type:
            return jsonify({"error": "Interview type not provided"}), 400

        conversation_history = []
        current_interview_type = interview_type
        if interview_type == "tech":
            tech_question_count = 0
            tech_score = 0
            question = generate_tech_question()
        else:
            hr_question_count = 0
            hr_score = 0
            hr_emotions_history = []
            hr_soft_skills_history = []
            question = generate_hr_question()

        audio_file = text_to_speech(question)
        if not audio_file:
            logger.warning("Audio generation failed, proceeding without audio")

        conversation_history.append({"role": "interviewer", "text": question})
        response = {"question": question, "audio": audio_file or None}
        logger.info(f"Start interview response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in start_interview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit_response', methods=['POST'])
def submit_response():
    global conversation_history, current_interview_type, tech_question_count, tech_score, hr_question_count, hr_score, hr_emotions_history, hr_soft_skills_history
    try:
        if not current_user_id:
            return jsonify({"error": "User not authenticated"}), 401

        interview_type = request.form.get('type') or current_interview_type
        if not interview_type:
            return jsonify({"error": "Interview type not provided"}), 400

        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        temp_audio_path = "temp_audio.webm"
        audio_file.save(temp_audio_path)
        audio_path = convert_to_wav(temp_audio_path)
        if not audio_path:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return jsonify({"error": "Failed to convert audio"}), 500
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        response_text = recognizer.recognize_google(audio, show_all=False) or "Could not understand audio."

        conversation_history.append({"role": "user", "text": response_text})

        pitch, energy = analyze_speech(audio_path)
        image_data = request.form.get('image_data')
        emotions, dominant_emotion = None, None
        if image_data:
            frame = capture_frame(image_data)
            if frame is not None:
                emotions, dominant_emotion = detect_emotion(frame)
            else:
                logger.warning("No frame captured from image data")

        soft_skills = analyze_soft_skills(response_text, pitch, energy, emotions)

        if interview_type == "tech":
            mark = gemini_mark_answer(response_text)
            tech_score += mark
            tech_question_count += 1
            if tech_question_count >= MAX_TECH_QUESTIONS:
                final_message = f"Tech Interview Completed. Your score is {tech_score} out of {MAX_TECH_QUESTIONS}."
                audio_file = text_to_speech(final_message)
                conversation_history.append({"role": "interviewer", "text": final_message})
                update_data = {
                    "tech_score": float(tech_score),
                    "last_updated": datetime.utcnow().isoformat()
                }
                logger.info(f"Attempting to update tech_score to {tech_score} for user_id: {current_user_id}")
                update_response = supabase.table("profiles").update(update_data).eq("user_id", current_user_id).execute()
                logger.info(f"Tech update response: {update_response.data}")
                profile_check = supabase.table("profiles").select("tech_score").eq("user_id", current_user_id).execute()
                if profile_check.data and profile_check.data[0]["tech_score"] == float(tech_score):
                    logger.info(f"Verified tech_score updated to {tech_score} for user_id: {current_user_id}")
                else:
                    logger.error(f"Failed to verify tech_score update for user_id: {current_user_id}. Current value: {profile_check.data}")
                response = {"question": final_message, "audio": audio_file or None}
            else:
                next_question = generate_tech_question(response_text)
                audio_file = text_to_speech(next_question)
                conversation_history.append({"role": "interviewer", "text": next_question})
                response = {"question": next_question, "audio": audio_file or None}
        else: 
            mark = gemini_mark_hr_answer(response_text)
            hr_score += mark
            hr_question_count += 1
            hr_emotions_history.append(emotions if emotions else {})
            hr_soft_skills_history.append(soft_skills)
            if hr_question_count >= MAX_HR_QUESTIONS:
                final_message = f"HR Interview Completed. Your score is {hr_score} out of {MAX_HR_QUESTIONS}. Check your profile for a detailed report."
                audio_file = text_to_speech(final_message)
                conversation_history.append({"role": "interviewer", "text": final_message})
                update_data = {
                    "hr_score": float(hr_score),
                    "hr_emotions": hr_emotions_history, 
                    "hr_soft_skills": hr_soft_skills_history, 
                    "last_updated": datetime.utcnow().isoformat()
                }
                logger.info(f"Attempting to update HR profile for user_id: {current_user_id}")
                update_response = supabase.table("profiles").update(update_data).eq("user_id", current_user_id).execute()
                logger.info(f"HR update response: {update_response.data}")
                response = {"question": final_message, "audio": audio_file or None}
            else:
                next_question = generate_hr_question()
                audio_file = text_to_speech(next_question)
                conversation_history.append({"role": "interviewer", "text": next_question})
                response = {
                    "question": next_question,
                    "audio": audio_file or None,
                    "emotions": emotions if emotions else {},
                    "dominant_emotion": dominant_emotion if dominant_emotion else "None"
                }

        logger.info(f"Submit response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in submit_response: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists("temp_audio.webm"):
            os.remove("temp_audio.webm")
        if os.path.exists("response.wav"):
            os.remove("response.wav")

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    global extracted_skills
    try:
        if not current_user_id:
            return jsonify({"error": "User not authenticated"}), 401

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

@app.route('/profile', methods=['GET'])
def profile():
    try:
        if not current_user_id:
            return jsonify({"error": "User not authenticated"}), 401

        profile_data = supabase.table("profiles").select("*").eq("user_id", current_user_id).execute()
        if not profile_data.data:
            return jsonify({"error": "Profile not found"}), 404

        profile = profile_data.data[0]
        logger.info(f"Fetched tech profile for user_id {current_user_id}: tech_score={profile['tech_score']}")
        return jsonify({
            "score": float(profile["tech_score"]), 
            "max_score": profile["tech_max_score"]
        })
    except Exception as e:
        logger.error(f"Error in profile route: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/hr_profile', methods=['GET'])
def hr_profile():
    try:
        if not current_user_id:
            return jsonify({"error": "User not authenticated"}), 401

        profile_data = supabase.table("profiles").select("*").eq("user_id", current_user_id).execute()
        if not profile_data.data:
            return jsonify({"error": "Profile not found"}), 404

        profile = profile_data.data[0]
        hr_score = float(profile["hr_score"])
        MAX_HR_QUESTIONS = profile["hr_max_score"]
        hr_emotions_history = profile["hr_emotions"]
        if isinstance(hr_emotions_history, str):
            hr_emotions_history = json.loads(hr_emotions_history) if hr_emotions_history else []
        hr_soft_skills_history = profile["hr_soft_skills"]
        if isinstance(hr_soft_skills_history, str):
            hr_soft_skills_history = json.loads(hr_soft_skills_history) if hr_soft_skills_history else []

        avg_emotions_description = ""
        avg_emotions = {}  
        if hr_emotions_history:
            for emotion_dict in hr_emotions_history:
                if emotion_dict:  
                    for emotion, value in emotion_dict.items():
                        avg_emotions[emotion] = avg_emotions.get(emotion, 0) + value
            if avg_emotions:
                for emotion in avg_emotions:
                    avg_emotions[emotion] /= len(hr_emotions_history)

                prominent_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:2]
                if prominent_emotions:
                    primary_emotion, primary_value = prominent_emotions[0]
                    if primary_value > 40:
                        avg_emotions_description += f"You came across as quite {primary_emotion} during the interview. "
                    elif primary_value > 20:
                        avg_emotions_description += f"You showed some {primary_emotion} vibes at times. "
                    else:
                        avg_emotions_description += f"You stayed pretty balanced, with a hint of {primary_emotion}. "
                    
                    if len(prominent_emotions) > 1:
                        secondary_emotion, secondary_value = prominent_emotions[1]
                        if secondary_value > 20:
                            avg_emotions_description += f"There was also a touch of {secondary_emotion} in your responses."
            else:
                avg_emotions_description = "Your emotions were pretty neutral throughout—nice and steady!"
        else:
            avg_emotions_description = "No emotional data available from your interview."

        confidence_count = {"High": 0, "Low": 0}
        enthusiasm_count = {"High": 0, "Low": 0}
        avg_positivity = 0
        for skills in hr_soft_skills_history:
            confidence_count[skills["confidence"]] += 1
            enthusiasm_count[skills["enthusiasm"]] += 1
            avg_positivity += skills["positivity"]
        avg_positivity /= len(hr_soft_skills_history) if hr_soft_skills_history else 1

        confidence_description = (
            "You sounded confident most of the time—great job keeping your voice steady!" 
            if confidence_count["High"] >= confidence_count["Low"] 
            else "You seemed a bit hesitant at times; try speaking up a little more next time."
        )
        enthusiasm_description = (
            "Your energy was infectious—you really brought some enthusiasm to the table!" 
            if enthusiasm_count["High"] >= enthusiasm_count["Low"] 
            else "You could perk up a bit; adding some energy might make your answers pop more."
        )
        positivity_description = (
            "Your responses had a nice positive vibe—very uplifting!" if avg_positivity > 0.2 
            else "You were fairly neutral; maybe sprinkle in some positivity to shine brighter!" if avg_positivity >= -0.2 
            else "Things felt a bit downbeat; try focusing on the brighter side in your answers."
        )

        feedback = []
        if hr_score < MAX_HR_QUESTIONS * 0.7:
            feedback.append("Your answers could use a bit more clarity and polish—try structuring them with a clear start, middle, and end.")
        if confidence_count["Low"] > confidence_count["High"]:
            feedback.append("You might want to practice speaking with more confidence; a louder, steady tone can make a big difference.")
        if enthusiasm_count["Low"] > enthusiasm_count["High"]:
            feedback.append("Bring some more enthusiasm to your voice—varying your tone can show you’re engaged and excited.")
        if avg_positivity < 0:
            feedback.append("Try to keep a positive spin on things—it helps leave a great impression!")
        if "angry" in avg_emotions and avg_emotions["angry"] > 20:
            feedback.append("You seemed a bit frustrated at times; staying calm and composed could help you come across even better.")

        overall_summary = f"Overall, you scored {hr_score} out of {MAX_HR_QUESTIONS}, which is "
        if hr_score >= MAX_HR_QUESTIONS * 0.9:
            overall_summary += "fantastic—you're really shining in these interviews! "
        elif hr_score >= MAX_HR_QUESTIONS * 0.7:
            overall_summary += "solid—you’re doing well with room to polish a few things. "
        else:
            overall_summary += "a good start—there’s definitely potential to build on! "

        areas_to_improve = []
        if hr_score < MAX_HR_QUESTIONS * 0.9:
            if confidence_count["Low"] > confidence_count["High"] or enthusiasm_count["Low"] > enthusiasm_count["High"]:
                areas_to_improve.append("working on your delivery—confidence and enthusiasm can really elevate your presence")
            if avg_positivity < 0.2:
                areas_to_improve.append("adding a bit more positivity to your tone—it can make you more memorable")
            if hr_score < MAX_HR_QUESTIONS * 0.7:
                areas_to_improve.append("structuring your answers more clearly—think about giving concise examples with impact")
            if not areas_to_improve:
                areas_to_improve.append("fine-tuning small details to push your performance to the next level")
        
        overall_summary += "To improve, focus on " + " and ".join(areas_to_improve) + ". Keep practicing, and you’ll get even stronger!"

        report = {
            "score": hr_score,
            "max_score": MAX_HR_QUESTIONS,
            "emotions": avg_emotions_description,
            "confidence": confidence_description,
            "enthusiasm": enthusiasm_description,
            "positivity": positivity_description,
            "feedback": feedback if feedback else ["You’re doing great—keep it up with consistent practice!"],
            "overall_summary": overall_summary
        }
        logger.info(f"HR profile report: {report}")
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error generating HR profile: {e}")
        return jsonify({"error": "Unable to generate report"}), 500

if __name__ == '__main__':
    app.run(debug=True)