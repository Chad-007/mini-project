import os
import time
import google.generativeai as genai
import speech_recognition as sr

genai.configure(api_key="AIzaSyAU3lqb-xlubVzrTyDslPaxX_tmUD1i_eo")
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_to_speech(timeout=5):
    with microphone as source:
        print("Listening for your answer...")
        recognizer.adjust_for_ambient_noise(source)
        
        start_time = time.time()
        try:
            audio = recognizer.listen(source, timeout=timeout)
            duration = time.time() - start_time
            print(f"Listened for {duration:.2f} seconds")
        except sr.WaitTimeoutError:
            print(f"Listening timed out after {timeout} seconds")
            return ""
        
    try:
        speech_text = recognizer.recognize_google(audio)
        print(f"You said: {speech_text}")
        return speech_text
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        return ""
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")
        return ""

def ai_interviewer(topic):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(
        history=[]
    )

    # Initial prompt to set up the interview context
    initial_prompt = (
        f"You are an AI interviewer conducting an interview about {topic}. "
        "Ask one question at a time and wait for the response. "
        "Make your questions engaging and conversational. "
        "After each answer, provide a brief acknowledgment or comment before asking the next question."
    )

    # Start the interview with the initial prompt
    print(f"\nStarting interview about {topic}...")
    response = chat_session.send_message(initial_prompt)
    print("\nAI Interviewer: " + response.text)

    while True:
        # Get speech input (answer) from the user
        user_input = listen_to_speech(timeout=7)  # 7 seconds timeout

        if not user_input:
            print("\nLet me repeat the question...")
            continue

        if "exit" in user_input.lower():
            print("\nExiting interview. Thank you for your time!")
            break

        # Send user response to Gemini and get follow-up
        response = chat_session.send_message(user_input)
        print("\nAI Interviewer: " + response.text)

def main():
    # List of available topics
    topics = [
        "Artificial Intelligence",
        "Climate Change",
        "Space Exploration",
        "Future of Technology",
        "Healthcare Innovation"
    ]
    
    # Print available topics
    print("Available Interview Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    # Get topic selection
    while True:
        try:
            choice = int(input("\nSelect a topic number (1-5): "))
            if 1 <= choice <= len(topics):
                selected_topic = topics[choice-1]
                break
            else:
                print("Please select a valid topic number.")
        except ValueError:
            print("Please enter a valid number.")

    # Start the interview with selected topic
    ai_interviewer(selected_topic)

if __name__ == "__main__":
    main()