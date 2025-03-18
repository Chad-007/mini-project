import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from gtts import gTTS
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Configure Gemini API
genai.configure(api_key="AIzaSyAU3lqb-xlubVzrTyDslPaxX_tmUD1i_eo")

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

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Store conversation history
conversation_history = []

# Function to detect facial emotion using DeepFace
def detect_emotion(frame):
    try:
        # Analyze the frame for emotions
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
        return None, None  # Always return 2 values even on error

# Function to capture video frame from the frontend
def capture_frame(image_data):
    try:
        # Check if image_data contains the base64 prefix
        if "," in image_data:
            # Decode base64 image data
            header, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
        else:
            # Try to decode directly if no prefix
            try:
                binary_data = base64.b64decode(image_data)
            except Exception as e:
                logger.error(f"Failed to decode image data: {e}")
                return None
        
        # Save to temp file
        temp_path = "temp_frame.jpg"
        with open(temp_path, "wb") as f:
            f.write(binary_data)
        
        # Read the image
        frame = cv2.imread(temp_path)
        
        # Check if image was read successfully
        if frame is None:
            logger.error("Failed to read image from file")
            return None
            
        return frame
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None

# Generate technical question
def generate_tech_question(response=None):
    if not model:
        logger.warning("Model not initialized, using fallback question")
        return "What is the difference between a list and a tuple in Python?"
    try:
        if not response:
            prompt = "Ask a beginner-level technical interview question about Python."
        else:
            prompt = f"Based on the response: '{response}', ask a follow-up technical question about Python."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}")
        return "What is the difference between a list and a tuple in Python?"

# Generate HR question
def generate_hr_question():
    return "Tell me about a time you worked in a team. How did you contribute?"

# Convert text to speech
def text_to_speech(text, filename="question.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        audio_path = os.path.join("static", filename)
        tts.save(audio_path)
        logger.info(f"Audio file saved at: {audio_path}")
        return filename
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
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
        
        # Add emotion analysis
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

# Route to start the interview
@app.route('/')
def index():
    return render_template('index.html')

# Route to start the interview
@app.route('/start_interview', methods=['POST'])
def start_interview():
    global conversation_history
    try:
        interview_type = request.json.get('type')
        if not interview_type:
            logger.error("Interview type not provided")
            return jsonify({"error": "Interview type not provided"}), 400

        conversation_history = []
        
        if interview_type == "tech":
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
        return jsonify({"error": str(e)}), 500

# Route to submit response
@app.route('/submit_response', methods=['POST'])
def submit_response():
    global conversation_history
    try:
        # Handle form data for interview_type and audio file
        interview_type = request.form.get('type')
        if not interview_type:
            logger.error("Interview type not provided")
            return jsonify({"error": "Interview type not provided"}), 400

        if 'audio' not in request.files:
            logger.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file with its original name temporarily
        temp_audio_path = os.path.join("temp_audio")
        audio_file.save(temp_audio_path)
        
        # Convert to WAV format
        audio_path = convert_to_wav(temp_audio_path)
        if not audio_path:
            return jsonify({"error": "Failed to convert audio to WAV"}), 500

        # Clean up temporary file
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
        
        # Capture and analyze facial emotion
        emotions = None
        dominant_emotion = None
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
                    emotions = None
                    dominant_emotion = Nones
            next_question = generate_tech_question(response_text)
            audio_file = text_to_speech(next_question)
            if not audio_file:
                return jsonify({"error": "Failed to generate audio"}), 500
            conversation_history.append({"role": "interviewer", "text": next_question})
            logger.info(f"Tech interview, next question: {next_question}")
            return jsonify({"question": next_question, "audio": audio_file})
        
        else:
            pitch, energy = analyze_speech(audio_path)
            soft_skills = analyze_soft_skills(response_text, pitch, energy, emotions)
            
            # Create feedback with proper handling for None values
            feedback_parts = [
                f"Confidence: {soft_skills['confidence']}",
                f"Enthusiasm: {soft_skills['enthusiasm']}",
                f"Positivity: {soft_skills['positivity']:.2f}"
            ]
            
            if soft_skills['emotion_feedback']:
                feedback_parts.append(soft_skills['emotion_feedback'])
                
            feedback = ", ".join(feedback_parts)
            
            next_question = "How do you handle stress in the workplace?"
            audio_file = text_to_speech(next_question)
            if not audio_file:
                return jsonify({"error": "Failed to generate audio"}), 500
            conversation_history.append({"role": "interviewer", "text": next_question})
            logger.info(f"HR interview, next question: {next_question}, feedback: {feedback}")
            return jsonify({"question": next_question, "audio": audio_file, "feedback": feedback})
    except Exception as e:
        logger.error(f"Error in submit_response: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 