import os
import time
import google.generativeai as genai
import speech_recognition as sr
import sys

# Suppress ALSA and JACK warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["ALSA_CONFIG_PATH"] = "/dev/null"
sys.stderr = open(os.devnull, "w")  # Redirect stderr to /dev/null

genai.configure(api_key="AIzaSyAU3lqb-xlubVzrTyDslPaxX_tmUD1i_eo")

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_to_speech(timeout=7):
    with microphone as source:
        print("\nListening for your answer... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        time.sleep(2)  # Allow time to process the question

        try:
            audio = recognizer.listen(source, timeout=timeout)
        except sr.WaitTimeoutError:
            print("\nNo response detected. Let's try again.")
            return ""

    try:
        speech_text = recognizer.recognize_google(audio)
        print(f"\nYou said: {speech_text}")
        return speech_text
    except sr.UnknownValueError:
        print("\nCouldn't understand. Please try again.")
        return ""
    except sr.RequestError:
        print("\nSpeech recognition service unavailable.")
        return ""

def ai_interviewer(topic):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel("gemini-2.0-flash-exp", generation_config=generation_config)
    chat_session = model.start_chat(history=[])

    initial_prompt = (
        f"You are an AI interviewer conducting an interview about {topic}. "
        "Ask engaging questions one at a time. Wait for the response before proceeding."
    )

    print(f"\nStarting interview about {topic}...")
    response = chat_session.send_message(initial_prompt)
    print("\nAI Interviewer:", response.text)

    while True:
        time.sleep(3)  # Short pause before listening
        user_input = listen_to_speech(timeout=10)

        if not user_input:
            print("\nRepeating the question...\n")
            continue

        if "exit" in user_input.lower():
            print("\nExiting interview. Thank you!")
            break

        response = chat_session.send_message(user_input)
        print("\nAI Interviewer:", response.text)

def main():
    topics = ["Artificial Intelligence", "Climate Change", "Space Exploration", "Future of Technology", "Healthcare Innovation"]
    
    print("Available Interview Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    while True:
        try:
            choice = int(input("\nSelect a topic number (1-5): "))
            if 1 <= choice <= len(topics):
                selected_topic = topics[choice - 1]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Enter a valid number.")

    ai_interviewer(selected_topic)

if __name__ == "__main__":
    main()
